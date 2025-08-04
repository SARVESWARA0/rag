import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import FireCrawlApp from '@mendable/firecrawl-js';
import { main as loaderMain } from './loader.js';
import { generateText, wrapLanguageModel, extractReasoningMiddleware, embed } from 'ai';
import { createMistral } from '@ai-sdk/mistral';
import { Pinecone } from '@pinecone-database/pinecone';

// Initialize Mistral model with reasoning middleware
const mistral = createMistral({ apiKey: process.env.MISTRAL_API_KEY });
const ragModel = wrapLanguageModel({
  model: mistral('mistral-large-latest'),
  middleware: extractReasoningMiddleware({ tagName: 'think' })
});


// Initialize Pinecone service
class PineconeService {
  constructor() {
    this.pinecone = null;
    this.index = null;
    this.initialize();
  }

  async initialize() {
    if (!this.pinecone) {
      this.pinecone = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
      });

      this.index = this.pinecone.index('rag');
    }
  }

  async findSources(text) {
    try {
      // Ensure services are initialized
      if (!this.index) {
        await this.initialize();
      }

      // Generate embedding using Mistral model
      const { embedding } = await embed({
        model: mistral.textEmbeddingModel('mistral-embed'),
        value: text,
      });

      // Query Pinecone for relevant contexts
      const queryResponse = await this.index.namespace('default').query({
        vector: embedding,
        topK: 16,
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
    console.log('Authorization token received:', token);
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
  maxAge: 14400000, // 4 hours cache
  timeout: 10000000, // ⬅️ Add this line: timeout in milliseconds (100s)
  
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
      console.error('Error details:', {
        message: scrapeError.message,
        code: scrapeError.code,
        status: scrapeError.status,
        url: documents
      });
      
      // Check if it's an authentication error
      if (scrapeError.message?.includes('InvalidAuthenticationInfo') || 
          scrapeError.message?.includes('Authentication information is not given')) {
        throw new Error('FireCrawl authentication failed. Please check your FIRECRAWL_API_KEY environment variable.');
      }
      
      throw new Error(`Failed to scrape document content: ${scrapeError.message}`);
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
           model: ragModel,
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
         
         // Handle specific model compatibility errors
         if (error.message.includes('Unsupported model version') || error.message.includes('AI_UnsupportedModelVersionError')) {
           answers.push('Error: Model compatibility issue. Please check AI SDK version and model configuration.');
         } else {
           answers.push(`Error generating answer: ${error.message}`);
         }
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




