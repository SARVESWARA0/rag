import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import FireCrawlApp from '@mendable/firecrawl-js';
import { main as loaderMain } from './loader.js';
import { generateText } from 'ai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenerativeAI } from '@google/generative-ai';

// Initialize Google Generative Model
const google = createGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_API_KEY
});

const model = google('gemini-2.0-flash');

// Initialize Pinecone service
class PineconeService {
  constructor() {
    this.pinecone = null;
    this.index = null;
    this.embedModel = null;
    this.initialize();
  }

  async initialize() {
    if (!this.pinecone) {
      this.pinecone = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
      });

      this.index = this.pinecone.index('rag');

      const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
      this.embedModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });
    }
  }

  async findSources(text) {
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
        topK: 15,
        includeMetadata: true
      });

      if (!queryResponse?.matches?.length) {
        console.log('No matches found');
        return [];
      }

      

      // Filter and return content from matches
      return queryResponse.matches.map(match => match.metadata?.content || '');
        

    } catch (error) {
      console.error('Error in findSources:', error);
      return [];
    }
  }
}

const pineconeService = new PineconeService();

// Initialize FireCrawl
const firecrawlApp = new FireCrawlApp({
  apiKey: process.env.FIRECRAWL_API_KEY
});

export async function GET(request) {
  return NextResponse.json({ 
    message: 'API is working! Use POST method with documents and questions.',
    status: 'ok'
  });
}

export async function POST(request) {
  try {
    console.log('POST request received');
    
    const body = await request.json();
    console.log('Request body:', body);
    
    const { documents, questions } = body;

    if (!documents || !questions) {
      return NextResponse.json({ 
        error: 'Missing required fields: documents and questions' 
      }, { status: 400 });
    }

    // Auth header validation
    const authHeader = request.headers.get('authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'Missing or invalid authorization header' }, { status: 401 });
    }
    const token = authHeader.split(' ')[1];

    console.log('Processing document URL:', documents);
    console.log('Questions count:', questions?.length || 0);

    // Use FireCrawl to scrape PDF content
    console.log('Starting FireCrawl PDF scraping...');
    let markdownContent = '';
    
    try {
      const scrapeResult = await firecrawlApp.scrapeUrl(documents, {
        formats: ["markdown"],
        onlyMainContent: true,
        parsePDF: true,
        maxAge: 14400000 // 4 hours cache
      });

      console.log('FireCrawl scraping completed successfully');
      console.log('Scraped content length:', scrapeResult.markdown?.length || 0);
      
      if (scrapeResult.markdown) {
        markdownContent = scrapeResult.markdown;
      } else {
        throw new Error('No markdown content returned from FireCrawl');
      }
      
    } catch (scrapeError) {
      console.error('FireCrawl scraping failed:', scrapeError);
      throw new Error('Failed to scrape document content');
    }

    // Clean up the markdown content
    markdownContent = markdownContent
      .replace(/\n{3,}/g, '\n\n') // Remove excessive newlines
      .replace(/\s+/g, ' ') // Normalize whitespace
      .trim();
    
    console.log('Markdown content prepared, length:', markdownContent.length);

    // Save markdown to data.txt
    const dataDir = path.join(process.cwd(), 'data');
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }
    const filePath = path.join(dataDir, 'data.txt');
    fs.writeFileSync(filePath, markdownContent, 'utf8');
    console.log('Document saved to:', filePath);

    // Upload to vector DB
    console.log('Starting vector database upload...');
    
    try {
      await loaderMain(filePath);
      console.log('Vector database upload completed successfully');
    } catch (err) {
      console.error('Vector database upload failed:', err);
      throw new Error(`Vector database upload failed: ${err.message}`);
    }

    // Sequentially generate answers for each question using direct RAG
    console.log('Generating answers for questions...');
    const answers = [];

    for (let i = 0; i < questions.length; i++) {
      const question = questions[i];
      console.log(`Processing question ${i + 1}/${questions.length}: ${question.substring(0, 100)}...`);
      
      try {
        // Retrieve relevant context from Pinecone
        const sources = await pineconeService.findSources(question);
        
        if (sources.length === 0) {
          console.log(`No relevant context found for question ${i + 1}`);
          answers.push("This information is not available in the provided policy document.");
          continue;
        }

        console.log(`Using ${sources.length} sources for context`);

        const context = sources
          .map((chunk, idx) => `[Context ${idx + 1}]: ${chunk}`)
          .join('\n\n');

                // Create enhanced system prompt with retrieved context
       
        const enhancedSystemPrompt = `
You are a Retrieval-Augmented Generation (RAG) chatbot designed to answer user questions strictly based on the provided insurance policy context.

INSTRUCTIONS:
1. Base your answers ONLY on the provided context from the policy document.
2. If the information is not available in the context, reply with: "This information is not available in the provided policy document."
3. Provide specific details, numbers, and exact policy terms when available.
4. Keep your answer concise (2–3 sentences) but comprehensive.
5. Include relevant policy clauses, sub-limits, waiting periods, and conditions.
6. Be precise with percentages, time periods, monetary limits, and coverage terms.
7. Focus on directly answering what the user asked, with no extra or assumed information.
Remember:read every line of the context carefully ,the answer will be in it so answer carefully.

CONTEXT:
${context}
Respond only using the above context.
`;
        const result = await generateText({ 
          model: model, 
          messages: [
            { role: 'system', content: enhancedSystemPrompt },
            { role: 'user', content: question }
          ],
          temperature: 0.1,
          maxTokens: 400
        });
        console.log(`Generated answer for question ${i + 1}:`, result.text.trim());
        answers.push(result.text.trim());
        
      } catch (error) {
        console.error(`Error generating answer for question ${i + 1}:`, error);
        answers.push(`Error generating answer: ${error.message}`);
      }
    }

    // Return comprehensive response
    return NextResponse.json({
      answers: answers,
      
    }, { status: 200 });

  } catch (error) {
    console.error('Error processing request:', error);
    return NextResponse.json({ 
      error: 'Internal Server Error', 
      details: error.message 
    }, { status: 500 });
  }
}




