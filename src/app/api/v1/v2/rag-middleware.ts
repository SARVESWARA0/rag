import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenerativeAI } from '@google/generative-ai';
import type {
  Experimental_LanguageModelV1Middleware as LanguageModelV1Middleware,
} from 'ai';

export class PineconeService {
  private pinecone: Pinecone | null = null;
  private index: any = null;
  private embedModel: any = null;

  constructor() {
    // Delay initialization to ensure async context
    this.initialize();
  }

  private async initialize() {
    if (!this.pinecone) {
      this.pinecone = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
      });

      this.index = this.pinecone.index('rag');

      const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
      this.embedModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });
    }
  }

  async findSources(text: string) {
    try {
      // Ensure services are initialized
      if (!this.embedModel || !this.index) {
        await this.initialize();
      }

      // Correctly format embedding request for a query
      const embeddingResult = await this.embedModel.embedContent({
        content: {
          parts: [{ text }]
        },
        taskType: 'RETRIEVAL_QUERY'
      });

      // Query Pinecone for relevant contexts
      const queryResponse = await this.index.namespace('default').query({
        vector: embeddingResult.embedding.values,
        topK: 2,
        includeMetadata: true
      });

      if (!queryResponse?.matches?.length) {
        console.log('No matches found');
        return [];
      }

      // Optional logging
      console.log(`Found ${queryResponse.matches.length} matches`);
      console.log('Preview:', queryResponse.matches[0].metadata.content?.substring(0, 200));

      // Filter and return content from matches
      return queryResponse.matches
        .filter(match => match.score > 1.0 && match.metadata.content)
        .map(match => match.metadata.content as string);

    } catch (error) {
      console.error('Error in findSources:', error);
      return [];
    }
  }
}

const pineconeService = new PineconeService();

function getLastUserMessageText({ prompt }: { prompt: any[] }) {
  const userMessages = prompt.filter(msg => msg.role === 'user');
  const lastMessage = userMessages.pop();

  if (!lastMessage?.content) return null;
  if (typeof lastMessage.content === 'string') return lastMessage.content;
  if (Array.isArray(lastMessage.content)) {
    return lastMessage.content
      .map(item => (typeof item === 'string' ? item : item?.text || ''))
      .join(' ');
  }
  return null;
}

export const pineconeMiddleware: LanguageModelV1Middleware = {
  transformParams: async ({ params }) => {
    try {
      const lastUserMessageText = getLastUserMessageText({ prompt: params.prompt });
      if (!lastUserMessageText) return params;

      const sources = await pineconeService.findSources(lastUserMessageText);
      if (!sources.length) return params;

      console.log(`Using ${sources.length} sources for context`);

      const context = sources
        .map((chunk, i) => `[Context ${i + 1}]: ${chunk}`)
        .join('\n\n');
      console.log('Context for RAG:', context);
      const enhancedSystemPrompt = `You are a RAG chatbo,you should respond only bassed on the content provided,Read the user query and data for RAG and provide your response note that the required answer will be provided in the  .

IMPORTANT INSTRUCTIONS:\n1. Base your answers ONLY on the provided context from the policy document\n2. If the information is not available in the context, clearly state "This information is not available in the provided policy document"\n3. Provide specific details, numbers, and exact policy terms when available\n4. Keep answers concise (2-3 sentences maximum) but comprehensive\n5. Include relevant policy clauses, waiting periods, and conditions\n6. Be precise with percentages, time periods, and coverage limits\n7. Focus on the most important information first\n
CONTEXT FROM POLICY DOCUMENT:\n${context}\n
Remember: Be concise, mainly answer what the user exactly needs in short, accurate, and reference specific policy terms from the provided context.`;

      return {
        ...params,
        prompt: [
          { role: 'system', content: enhancedSystemPrompt },
          ...params.prompt
        ]
      };
    } catch (error) {
      console.error('Error in transformParams:', error);
      return params;
    }
  },

  wrapGenerate: async ({ doGenerate }) => {
    try {
      const result = await doGenerate();
      if (result && 'text' in result) {
        console.log('Generated text:', result.text);
      }
      return result;
    } catch (error) {
      console.error('Error in wrapGenerate:', error);
      return await doGenerate();
    }
  }
};
