import { chromium, Page } from 'playwright';

/**
 * MCP to perform a web search using DuckDuckGo and scrape the results.
 * @param query The search query from the AI agent.
 * @returns A formatted string of search results.
 */
export async function searchTheWeb(query: string): Promise<string> {
  console.log(`[Browser MCP] Launching browser to search for: ${query}`);
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  try {
    // Navigate to DuckDuckGo with the search query
    await page.goto(`https://duckduckgo.com/?q=${encodeURIComponent(query)}&ia=web`);
    
    // Wait for the search results container to be visible
    await page.waitForSelector('#links', { timeout: 10000 });

    // Scrape the titles, links, and snippets of the top search results
    const results = await page.evaluate(() => {
        const items = Array.from(document.querySelectorAll('[data-testid="result"]')).slice(0, 4); // Get top 4 results
        return items.map(item => {
            const title = (item.querySelector('h2 a span') as HTMLElement)?.innerText;
            const link = (item.querySelector('h2 a') as HTMLAnchorElement)?.href;
            const snippet = (item.querySelector('[data-testid="result-snippet"]') as HTMLElement)?.innerText;
            return { title, link, snippet };
        });
    });

    await browser.close();

    if (results.length === 0) {
      return "No search results found. The page might have changed its structure.";
    }

    // Format the results into a single string for the AI agent to process
    const formattedResults = results
      .map((res, i) => `Result ${i + 1}:\nTitle: ${res.title}\nSnippet: ${res.snippet}\nURL: ${res.link}`)
      .join('\n\n---\n\n');
      
    console.log(`[Browser MCP] Scraped ${results.length} results.`);
    return formattedResults;

  } catch (error) {
    console.error(`[Browser MCP] Failed to search for "${query}":`, error);
    await browser.close();
    return `Error: Could not perform the web search. The search engine page might be blocking the request or has changed.`;
  }
}


/**
 * MCP for browsing a specific web page. This remains for potential future use.
 * @param url The URL of the webpage to scrape.
 * @returns The text content of the page body.
 */
export async function browseWebPage(url: string): Promise<string> {
  console.log(`[Browser MCP] Launching browser to visit: ${url}`);
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();
  
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    const content = await page.evaluate(() => document.body.innerText);
    console.log(`[Browser MCP] Successfully scraped ${content.length} characters from the page.`);
    await browser.close();
    return content.substring(0, 8000); 
  } catch (error) {
    console.error(`[Browser MCP] Failed to browse ${url}:`, error);
    await browser.close();
    return `Error: Could not access or read the content from the URL: ${url}.`;
  }
}
