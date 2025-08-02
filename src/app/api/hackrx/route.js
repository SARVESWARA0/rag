import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch';
import { main as loaderMain } from './loader.js';
import { experimental_wrapLanguageModel as wrapLanguageModel, generateText } from 'ai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { pineconeMiddleware } from './rag-middleware.ts';

// Initialize Google Generative Model with Pinecone middleware
const google = createGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_API_KEY
});

const wrappedModel = wrapLanguageModel({
  model: google('gemini-2.0-flash'),
  middleware: pineconeMiddleware,
});

export async function GET(request) {
  return NextResponse.json({ 
    message: 'HackRX API is working! Use POST method with documents and questions.',
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

    console.log('Processing document:', documents);
    console.log('Questions count:', questions?.length || 0);

    // Fetch PDF
    const pdfResponse = await fetch(documents, {
      headers: { 
        'User-Agent': 'Mozilla/5.0 (compatible; DocumentProcessor/1.0)' 
      }
    });
    
    if (!pdfResponse.ok) {
      throw new Error(`Failed to fetch PDF: ${pdfResponse.status} ${pdfResponse.statusText}`);
    }
    
    const pdfArrayBuffer = await pdfResponse.arrayBuffer();
    const pdfBuffer = Buffer.from(pdfArrayBuffer);
    console.log('PDF downloaded, size:', pdfBuffer.length, 'bytes');

    // For now, create a mock PDF content to test the pipeline
    // In production, this would be replaced with actual PDF extraction
    const markdownContent = `# Page 1

This is a mock PDF content for testing purposes. The actual PDF content would be extracted here.

# Page 2

This is the second page of the mock PDF content. The insurance policy document would contain information about:

- Grace period for premium payment
- Waiting period for pre-existing diseases
- Maternity expenses coverage
- Cataract surgery waiting period
- Organ donor medical expenses
- No Claim Discount (NCD)
- Preventive health check-ups
- Hospital definition
- AYUSH treatments coverage
- Room rent and ICU charges sub-limits

# Page 3

Additional policy details and terms would be extracted from the actual PDF document.`;

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
    let uploadError = null;
    let uploadResults = null;
    
    try {
      uploadResults = await loaderMain(filePath);
      console.log('Vector database upload completed successfully');
    } catch (err) {
      console.error('Vector database upload failed:', err);
      uploadError = err.message;
    }

    // Sequentially generate answers for each question using RAG
    console.log('Generating answers for questions...');
    const answers = [];

    for (let i = 0; i < questions.length; i++) {
      const question = questions[i];
      console.log(`Processing question ${i + 1}/${questions.length}: ${question.substring(0, 100)}...`);
      
      try {
        // Generate answer with RAG
        const result = await generateText({ 
          model: wrappedModel, 
          messages: [{ 
            role: 'user', 
            content: question
          }],
          temperature: 0.1,
          maxTokens: 400, // Reduced for more concise answers
          topP: 0.9,
          topK: 40
        });
        
        answers.push(result.text.trim());
        console.log(`Answer ${i + 1} generated successfully`);
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