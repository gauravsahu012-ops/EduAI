// chat.js

import * as dotenv from 'dotenv';
dotenv.config();

import readlineSync from 'readline-sync';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: "", // store in .env
});

const History = []; // conversation history

// Rewriter function using Grok model
async function transformQuery(question) {
  const transformHistory = [...History];
  transformHistory.push({
    role: "user",
    content: `Based on our chat history, rephrase this follow-up question into a complete, standalone question: "${question}"`,
  });

  const completion = await client.chat.completions.create({
    model: "x-ai/grok-4-fast:free",
    messages: transformHistory,
    extra_headers: {
      "HTTP-Referer": "<YOUR_SITE_URL>", // optional
      "X-Title": "<YOUR_SITE_NAME>",     // optional
    },
  });

  return completion.choices[0].message.content.trim();
}

// Main RAG loop
async function chatting(question) {
  try {
    // 1. Rewrite query
    const rewrittenQuery = await transformQuery(question);
    console.log(`\n(Rewritten Query: ${rewrittenQuery})`);

    // 2. Get embeddings
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: "text-embedding-004",
    });
    const queryVector = await embeddings.embedQuery(rewrittenQuery);

    // 3. Pinecone search
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    const searchResults = await pineconeIndex.query({
      topK: 10,
      vector: queryVector,
      includeMetadata: true,
    });

    // 4. Build context
    const context = searchResults.matches
      .map((m) => m.metadata.text)
      .join("\n\n---\n\n");

    // 5. Create final RAG prompt
    const finalPrompt = `
Context: ${context}
---
Question: ${rewrittenQuery}
`;

    // push user query with RAG context
    History.push({
      role: "user",
      content: finalPrompt,
    });

    // 6. Get answer from Grok (via OpenRouter)
    const response = await client.chat.completions.create({
      model: "x-ai/grok-4-fast:free",
      messages: History,
      extra_headers: {
        "HTTP-Referer": "<YOUR_SITE_URL>",
        "X-Title": "<YOUR_SITE_NAME>",
      },
    });

    const finalAnswer = response.choices[0].message.content.trim();

    // save answer to history
    History.push({
      role: "assistant",
      content: finalAnswer,
    });

    console.log("\n");
    console.log(finalAnswer);
  } catch (error) {
    console.error("An error occurred during chat:", error);
  }
}

// Run chatbot loop
async function main() {
  console.log("Welcome to the DSA Chatbot! Type 'exit' to quit.");
  while (true) {
    const userProblem = readlineSync.question("Ask me anything--> ");
    if (userProblem.toLowerCase() === "exit") {
      console.log("Goodbye!");
      break;
    }
    await chatting(userProblem);
  }
}

main();
