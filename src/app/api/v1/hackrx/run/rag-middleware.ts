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
    this.initialize();
  }

  private async initialize() {
    if (!this.pinecone) {
      this.pinecone = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY ,
      });

      this.index = this.pinecone.index('rag');
      
      const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
      this.embedModel = genAI.getGenerativeModel({ model: "text-embedding-004" });
    }
  }

  async findSources(text: string) {
    try {
      if (!this.embedModel || !this.index) {
        await this.initialize();
      }

      const embeddingResult = await this.embedModel.embedContent(text);
      
      

      const queryResponse = await this.index.namespace('default').query({
        vector: embeddingResult.embedding.values,
        topK: 3,
        includeMetadata: true,
        
      });
      console.log('Query response:', queryResponse.matches[0].metadata.content);
      if (!queryResponse?.matches) {
        console.log('No matches found in query response');
        return [];
      }

      return queryResponse.matches
        .filter(match => match?.metadata?.content)
        .map(match => match.metadata.content);

    } catch (error) {
      console.error('Error in findSources:', error);
      return [];
    }
  }
}

const pineconeService = new PineconeService();

function getLastUserMessageText({ prompt }: { prompt: any[] }) {
  const userMessages = prompt.filter(msg => msg.role === 'user');
  const lastMessage = userMessages[userMessages.length - 1];
  
  if (!lastMessage?.content) return null;
  
  if (typeof lastMessage.content === 'string') {
    return lastMessage.content;
  }
  
  if (Array.isArray(lastMessage.content)) {
    return lastMessage.content
      .map(item => typeof item === 'string' ? item : item?.text || '')
      .join(' ');
  }
  
  return null;
}

export const pineconeMiddleware: LanguageModelV1Middleware = {
  transformParams: async ({ params }) => {
    try {
      const lastUserMessageText = getLastUserMessageText({ prompt: params.prompt });
      if (!lastUserMessageText) {
        console.log('No user message found');
        return params;
      }

      const sources = await pineconeService.findSources(lastUserMessageText);
      if (!sources.length) {
        console.log('No sources found');
        return params;
      }

      const context = sources.map(chunk => JSON.stringify(chunk)).join('\n');

      return {
        ...params,
        prompt: [
          {
            role: 'system',
            content: `Use this information to answer the question:\n${context}`
          },
          ...params.prompt
        ]
      };
    } catch (error) {
      console.error('Error in transformParams:', error);
      return params;
    }
  },

  wrapGenerate: async ({ doGenerate, params, model }) => {
    try {
      const result = await doGenerate();

      if (result && 'text' in result) {
        console.log('Generated text:', result.text);
      }

      return result;
    } catch (error) {
      console.error('Error in wrapGenerate middleware:', error);
      return await doGenerate();
    }
  }
};
