import { NextRequest, NextResponse } from 'next/server';
import { searchTheWeb } from '@/mcp/browser'; // Import the browser MCP function

/**
 * This API route is called by the FastAPI agent when it decides to use the "WebSearch" tool.
 */
export async function POST(req: NextRequest) {
  try {
    const { query } = await req.json();

    if (!query) {
      return NextResponse.json({ error: 'Search query is required.' }, { status: 400 });
    }

    console.log(`[Search MCP Endpoint] Received search request for: "${query}"`);
    const searchResults = await searchTheWeb(query);
    
    return NextResponse.json({ results: searchResults });

  } catch (error: any) {
    console.error('[Search MCP Endpoint] Error:', error.message);
    return NextResponse.json({ error: 'Failed to perform web search.' }, { status: 500 });
  }
}