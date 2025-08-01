import { pineconeMiddleware } from '../hackrx/run/rag-middleware';
import { experimental_wrapLanguageModel as wrapLanguageModel, generateText } from 'ai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';

const google = createGoogleGenerativeAI({
  apiKey: 'AIzaSyDd0ktqwKnFOfaQCU0dryXuhcnhiuybXFQ'
});

const wrappedModel = wrapLanguageModel({
  model: google('gemini-2.5-pro'),
  middleware: pineconeMiddleware,
});

export async function POST(req) {
  try {
    const { messages } = await req.json();

    const result = await generateText({
      model: wrappedModel,
      messages,
    });

    return new Response(JSON.stringify({ text: result.text }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Error in route handler:', error);
    return new Response(JSON.stringify({ error: 'Internal Server Error' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
