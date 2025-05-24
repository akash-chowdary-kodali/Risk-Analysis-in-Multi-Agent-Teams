// Use import syntax for ES Modules
import { chromium } from 'playwright';
import path from 'path';
import fs from 'fs'; // For file system operations like ensuring directory exists
import { fileURLToPath } from 'url'; // To get __dirname equivalent in ESM
import { dirname } from 'path'; // To get __dirname equivalent in ESM

// --- Configuration ---

// Get current directory path in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Agent values derived from HTML snippet provided by user
const AGENT_VALUES = {
    "Human-aware PPO": "ppo_bc",
    "Population-Based Training": "pbt",
    "Self-Play": "ppo_sp",
    "Human Keyboard Input": "human"
};
const AGENT_NAMES = Object.keys(AGENT_VALUES);

const LAYOUT_NAME = "Cramped Room";
// Layout value derived from HTML snippet provided by user
const LAYOUT_VALUE = "cramped_room";

const GAME_LENGTH_SEC = 60;
const NUM_RUNS_PER_PAIR = 20; // Set back to 20 for full run
const BASE_OUTPUT_DIR = "game_trajectories"; // Base directory for all trajectories
// Specific output directory for this layout
const LAYOUT_OUTPUT_DIR = path.join(BASE_OUTPUT_DIR, LAYOUT_NAME);

// --- Main Async Function ---
async function runExperiments() {
    console.log("Launching browser...");
    // Launch non-headless AND add slowMo to see actions clearly
    const browser = await chromium.launch({
        headless: false,
        slowMo: 500 // Delay (in ms) between Playwright actions. Adjust as needed.
    });
    const context = await browser.newContext({ acceptDownloads: true });
    const page = await context.newPage();

    console.log(`Ensuring output directory exists: ${LAYOUT_OUTPUT_DIR}`);
    fs.mkdirSync(LAYOUT_OUTPUT_DIR, { recursive: true });

    const agentPairs = [];
    for (let i = 0; i < AGENT_NAMES.length; i++) {
        for (let j = i; j < AGENT_NAMES.length; j++) {
            if (AGENT_NAMES[i] === "Human Keyboard Input" && AGENT_NAMES[j] === "Human Keyboard Input") {
                continue;
            }
            agentPairs.push([AGENT_NAMES[i], AGENT_NAMES[j]]);
        }
    }

    let actual_run_count = 0;
    let total_runs = 0;
    agentPairs.forEach(([a1, a2]) => {
        total_runs += NUM_RUNS_PER_PAIR;
        if (a1 !== a2) {
            total_runs += NUM_RUNS_PER_PAIR;
        }
    });

    console.log(`Starting experiment runs. Total runs planned: ${total_runs}`);

    for (const [agent1_name, agent2_name] of agentPairs) {
        const agent1_value = AGENT_VALUES[agent1_name];
        const agent2_value = AGENT_VALUES[agent2_name];

        // Run A vs B
        console.log(`\n=== Running Pair: ${agent1_name} vs ${agent2_name} ===`);
        for (let run_num = 1; run_num <= NUM_RUNS_PER_PAIR; run_num++) {
            actual_run_count++;
            console.log(`\n--- Run ${run_num}/${NUM_RUNS_PER_PAIR} (${agent1_name} vs ${agent2_name}) (Overall: ${actual_run_count}/${total_runs}) ---`);
            const baseFilename = `${agent1_name} vs ${agent2_name} ${GAME_LENGTH_SEC} sec ${LAYOUT_NAME} (${run_num}).json`;
            const targetFilePath = path.join(LAYOUT_OUTPUT_DIR, baseFilename);
            await runSingleGame(page, agent1_name, agent2_name, agent1_value, agent2_value, LAYOUT_VALUE, GAME_LENGTH_SEC, targetFilePath);
        }

        // Run B vs A if different agents
        if (agent1_name !== agent2_name) {
            const agent1_value_rev = AGENT_VALUES[agent2_name];
            const agent2_value_rev = AGENT_VALUES[agent1_name];
            console.log(`\n=== Running Pair: ${agent2_name} vs ${agent1_name} ===`);
            for (let run_num = 1; run_num <= NUM_RUNS_PER_PAIR; run_num++) {
                 actual_run_count++;
                 console.log(`\n--- Run ${run_num}/${NUM_RUNS_PER_PAIR} (${agent2_name} vs ${agent1_name}) (Overall: ${actual_run_count}/${total_runs}) ---`);
                 const baseFilename = `${agent2_name} vs ${agent1_name} ${GAME_LENGTH_SEC} sec ${LAYOUT_NAME} (${run_num}).json`;
                 const targetFilePath = path.join(LAYOUT_OUTPUT_DIR, baseFilename);
                 await runSingleGame(page, agent2_name, agent1_name, agent1_value_rev, agent2_value_rev, LAYOUT_VALUE, GAME_LENGTH_SEC, targetFilePath);
            }
        }
    }

    console.log("\nClosing browser...");
    await browser.close();
    console.log(`\n=== All ${actual_run_count} experiment runs finished. ===`);
    console.log(`Trajectories saved in: ${LAYOUT_OUTPUT_DIR}`);
}

// --- Helper function to run a single game ---
async function runSingleGame(page, agent1_name, agent2_name, agent1_value, agent2_value, layout_value, game_length, target_filepath) {
    if (agent1_name === "Human Keyboard Input" && agent2_name === "Human Keyboard Input") {
        console.log(`  Skipping run: Human vs Human is not supported/required.`);
        return;
    }
    if (fs.existsSync(target_filepath)) {
        console.log(`  Skipping run, file already exists: ${target_filepath}`);
        return;
    }

    console.log(`  Navigating to demo page...`);
    try {
        await page.goto('https://humancompatibleai.github.io/overcooked-demo/');
        // Use the correct layout selector ID based on HTML
        const initialElementSelector = '#layout';
        console.log(`    Waiting for selector: ${initialElementSelector}`);
        await page.waitForSelector(initialElementSelector, { state: 'visible', timeout: 15000 });
        console.log(`  Page loaded. Configuring game...`);

        // --- Using Corrected Selectors Based on Provided HTML ---
        const layoutSelector = '#layout'; // Correct ID from HTML
        const player0Selector = '#playerZero'; // Correct ID for Player 1 from HTML
        const player1Selector = '#playerOne'; // Correct ID for Player 2 from HTML
        const gameLengthSelector = '#gameTime'; // Correct ID for Game Length from HTML
        const saveTrajSelector = '#saveTrajectories'; // Correct ID for Save Checkbox from HTML
        // Start button click is removed as 'Enter' is the trigger

        console.log(`    Interaction Block - Using Selectors: Layout=${layoutSelector}, Player0=${player0Selector}, Player1=${player1Selector}, GameTime=${gameLengthSelector}, Save=${saveTrajSelector}`);

        // Select Layout
        console.log(`    Selecting layout: ${layout_value} using selector ${layoutSelector}`);
        await page.selectOption(layoutSelector, { value: layout_value });
        await page.waitForTimeout(100);

        // Select Agents - Assigning values based on Player 1/Player 2 roles
        // Agent 1 in the script corresponds to Player 1 (playerZero) on the website
        // Agent 2 in the script corresponds to Player 2 (playerOne) on the website
        console.log(`    Selecting Player 1 (agent1): ${agent1_value} using selector ${player0Selector}`);
        await page.selectOption(player0Selector, { value: agent1_value });
        await page.waitForTimeout(100);
        console.log(`    Selecting Player 2 (agent2): ${agent2_value} using selector ${player1Selector}`);
        await page.selectOption(player1Selector, { value: agent2_value });
        await page.waitForTimeout(100);

        // Set Game Length
        console.log(`    Setting game length: ${game_length} using selector ${gameLengthSelector}`);
        await page.locator(gameLengthSelector).clear(); // Clear field first
        await page.fill(gameLengthSelector, String(game_length));
        await page.waitForTimeout(100);

        // Check Save Trajectory
        console.log(`    Ensuring 'Save Trajectories' is checked using selector ${saveTrajSelector}`);
        await page.check(saveTrajSelector); // Ensures it's checked
        await page.waitForTimeout(100);

        // <<< Press Enter after configurations >>>
        console.log(`    Clicking body to potentially remove focus from inputs...`);
        await page.locator('body').click({force: true});
        await page.waitForTimeout(100);
        console.log(`    Pressing Enter key to start game...`);
        await page.keyboard.press('Enter');
        // No need to wait long here, the download listener starts immediately

        // Prepare for download BEFORE pressing Enter (or just after)
        console.log(`    Setting up download listener...`);
        const downloadTimeout = game_length * 1000 + 30000; // Increased buffer to 30s
        const downloadPromise = page.waitForEvent('download', { timeout: downloadTimeout });

        // Handle Human Input Case - Log message
        if (agent1_name === "Human Keyboard Input" || agent2_name === "Human Keyboard Input") {
             console.log("!!! HUMAN INPUT REQUIRED !!!");
             console.log(`!!! Please play the game in the browser window for ${game_length} seconds.`);
             console.log("!!! Waiting for game to finish and download to start...");
        } else {
            console.log(`    AI vs AI game running... waiting for download...`);
        }

        console.log(`    Waiting for download event (max ${downloadTimeout / 1000}s)...`);
        const download = await downloadPromise;
        console.log(`    Download detected: ${download.suggestedFilename()}`);

        // Save the download
        console.log(`    Saving download to: ${target_filepath}`);
        await download.saveAs(target_filepath);
        console.log(`    Trajectory saved successfully.`);

        // Add a small delay before the next interaction/navigation
        await page.waitForTimeout(2000);

    } catch (error) {
        console.error(`!!! Error during game run for target: ${target_filepath}`);
        console.error(`!!! Error details:`, error); // Log the full error
        console.error(`!!! Page URL at time of error: ${page.url()}`); // Log current URL
        // Try saving a screenshot for debugging
        const screenshotPath = target_filepath.replace('.json', '_error.png');
        try {
            await page.screenshot({ path: screenshotPath, fullPage: true });
            console.error(`!!! Screenshot saved to: ${screenshotPath}`);
        } catch (ssError) {
            console.error(`!!! Failed to save screenshot: ${ssError}`);
        }
        // Wait longer on error to allow inspection of the browser window
        console.error("!!! Pausing for 10 seconds after error...");
        await page.waitForTimeout(10000);
    }
     console.log(`  Finished single game run attempt for: ${target_filepath}`);
}


// --- Run the main function ---
runExperiments().catch(err => {
    console.error("Script failed:", err);
    process.exit(1); // Exit with error code
});
