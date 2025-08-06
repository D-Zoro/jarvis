import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';
// We no longer need the browser MCP here directly for the CLI
// import { browseWebPage } from '../../../mcp/browser'; 

const AI_BACKEND_URL = process.env.AI_BACKEND_URL || 'http://ai-fastapi:8001';

interface CliRequestBody {
  prompt: string;
  session_id: string;
}

/**
 * This is the main API route that the CLI interacts with.
 * It parses the user's input for special commands or forwards to the agent.
 */
export async function POST(req: NextRequest) {
  try {
    const { prompt, session_id } = (await req.json()) as CliRequestBody;

    // --- Command Parsing Logic for Memory ---
    if (prompt.toLowerCase().startsWith('remember this:')) {
      const memoryText = prompt.substring('remember this:'.length).trim();
      await axios.post(`${AI_BACKEND_URL}/memory/add`, { text: memoryText, session_id });
      return NextResponse.json({ response: `OK, I'll remember that.` });
    }

    if (prompt.toLowerCase().startsWith('forget everything')) {
      await axios.post(`${AI_BACKEND_URL}/memory/clear`, { session_id });
      return NextResponse.json({ response: `OK, I've cleared my memory for this session.` });
    }
    
    if (prompt.toLowerCase() === 'what did i say?') {
        const { data } = await axios.post(`${AI_BACKEND_URL}/memory/list`, { session_id });
        const memories = data.memories.length > 0 ? data.memories.join('\n- ') : 'Nothing yet.';
        return NextResponse.json({ response: `Here's what I remember:\n- ${memories}` });
    }

    // REMOVED: The 'browse' command is now handled automatically by the agent.
    // The agent will decide when to search the web based on the prompt.

    // --- Default Prompt Handling ---
    // If no command is matched, send the prompt directly to the AI agent.
    const { data } = await axios.post(`${AI_BACKEND_URL}/prompt`, { prompt, session_id });
    return NextResponse.json({ response: data.response });

  } catch (error: any) {
    console.error('Error in CLI API route:', error.message);
    if (error.code === 'ECONNREFUSED') {
        return NextResponse.json({ response: "Error: Could not connect to the AI backend. Is it running?" }, { status: 500 });
    }
    return NextResponse.json({ response: 'An error occurred on the server.' }, { status: 500 });
  }
}
