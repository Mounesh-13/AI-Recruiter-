// web/app.js

// --- Tab Navigation Logic ---
const tabs = document.querySelectorAll('.tab-btn');
const contents = document.querySelectorAll('.tab-content');

function changeTab(tabName) {
    tabs.forEach(tab => {
        if (tab.getAttribute('onclick').includes(tabName)) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });
    contents.forEach(content => {
        if (content.id === tabName) {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });
}

// --- Helper Functions ---
function showLoader(loaderId) {
    document.getElementById(loaderId).style.display = 'flex';
}

function hideLoader(loaderId) {
    document.getElementById(loaderId).style.display = 'none';
}

function displayError(outputId, message) {
    const outputDiv = document.getElementById(outputId);
    outputDiv.innerHTML = `<div class="text-red-600 bg-red-100 p-4 rounded-md">${message}</div>`;
}

// --- File Reading Helper ---
function readFileAsBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            // result includes the mime type header, so we split it
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = error => reject(error);
        reader.readAsDataURL(file);
    });
}


// --- Feature 1: Keyword Scoring ---
async function handleScoring() {
    const keywords = document.getElementById('keywords-scoring').value;
    const resumeFile = document.getElementById('resume-scoring').files[0];
    const outputDiv = document.getElementById('scoring-output');
    
    if (!keywords || !resumeFile) {
        displayError('scoring-output', 'Please provide both keywords and a resume file.');
        return;
    }

    showLoader('scoring-loader');
    outputDiv.innerHTML = '';

    try {
        const fileData = await readFileAsBase64(resumeFile);
        const fileInfo = { name: resumeFile.name, data: fileData };
        
        const result = await eel.process_resume_for_scoring(fileInfo, keywords)();
        
        hideLoader('scoring-loader');
        if (result.error) {
            displayError('scoring-output', result.error);
        } else {
            renderScoringResult(result);
        }
    } catch (error) {
        hideLoader('scoring-loader');
        displayError('scoring-output', `An error occurred: ${error}`);
    }
}

function renderScoringResult(result) {
    const outputDiv = document.getElementById('scoring-output');
    let breakdownHtml = '<p class="text-slate-500">No keywords found.</p>';

    if (result.breakdown && result.breakdown.length > 0) {
        breakdownHtml = `
            <ul class="space-y-3">
                ${result.breakdown.map(item => `
                    <li class="p-3 bg-slate-50 rounded-md">
                        <div class="flex justify-between items-center">
                            <span class="font-semibold text-indigo-700">${item.keyword}</span>
                            <span class="text-sm font-medium bg-indigo-100 text-indigo-800 px-2 py-1 rounded-full">Score: ${item.score}</span>
                        </div>
                        <div class="text-xs text-slate-500 mt-1">
                            Found ${item.count} times in: ${item.locations.join(', ') || 'N/A'}
                        </div>
                    </li>
                `).join('')}
            </ul>
        `;
    }

    outputDiv.innerHTML = `
        <div class="text-center mb-4">
            <span class="text-5xl font-bold text-indigo-600">${result.total_score}%</span>
            <p class="text-slate-600 font-medium">Overall Match Score</p>
        </div>
        <h3 class="text-lg font-semibold mt-6 mb-2">Keyword Breakdown</h3>
        ${breakdownHtml}
    `;
}


// --- Feature 2: Resume Ranking ---
async function handleRanking() {
    const keywords = document.getElementById('keywords-ranking').value;
    const resumeFiles = document.getElementById('resumes-ranking').files;
    const outputDiv = document.getElementById('ranking-output');

    if (!keywords || resumeFiles.length === 0) {
        displayError('ranking-output', 'Please provide keywords and at least one resume file.');
        return;
    }

    showLoader('ranking-loader');
    outputDiv.innerHTML = '';

    try {
        const fileList = [];
        for (const file of resumeFiles) {
            const fileData = await readFileAsBase64(file);
            fileList.push({ name: file.name, data: fileData });
        }

        const result = await eel.rank_resumes(fileList, keywords)();
        
        hideLoader('ranking-loader');
        if (result.error) {
            displayError('ranking-output', result.error);
        } else {
            renderRankingResult(result.ranking);
        }
    } catch (error) {
        hideLoader('ranking-loader');
        displayError('ranking-output', `An error occurred: ${error}`);
    }
}

function renderRankingResult(ranking) {
    const outputDiv = document.getElementById('ranking-output');
    if (!ranking || ranking.length === 0) {
        outputDiv.innerHTML = '<p class="text-slate-500">No results to display.</p>';
        return;
    }

    outputDiv.innerHTML = `
        <div class="overflow-x-auto">
            <table class="min-w-full divide-y divide-slate-200">
                <thead class="bg-slate-50">
                    <tr>
                        <th class="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Rank</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Candidate</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Final Score</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Details</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Shortlist?</th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-slate-200">
                    ${ranking.map((r, index) => `
                        <tr>
                            <td class="px-6 py-4 whitespace-nowrap text-sm font-bold text-indigo-600">${index + 1}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900">${r.filename}</td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm text-slate-500 font-semibold">${r.final_score}</td>
                            <td class="px-6 py-4 text-sm text-slate-500">
                                Kw Match: ${r.keyword_match}% | Exp: ${r.experience} yrs | Skills: ${r.skill_count} | Certs: ${r.certifications}
                                <p class="text-xs text-slate-400 mt-1 italic">Rationale: ${r.rationale}</p>
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm">
                                <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${r.shortlist === 'Yes' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}">
                                    ${r.shortlist}
                                </span>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
}

// --- Feature 3: Job Fit Q&A ---
async function handleQnA() {
    const jd = document.getElementById('qna-jd').value;
    const resume = document.getElementById('qna-resume').value;
    const questionInput = document.getElementById('qna-question');
    const question = questionInput.value;
    const chatbox = document.getElementById('qna-chatbox');

    if (!jd || !resume || !question) {
        alert('Please fill in the Job Description, Resume, and your Question.');
        return;
    }

    // Add user's question to chat
    chatbox.innerHTML += `<div class="text-right mb-2"><span class="bg-indigo-500 text-white rounded-lg py-2 px-4 inline-block">${question}</span></div>`;

    const response = await eel.process_qna(jd, resume, question)();
    
    // Add bot's response
    chatbox.innerHTML += `<div class="text-left mb-2"><span class="bg-slate-200 text-slate-800 rounded-lg py-2 px-4 inline-block whitespace-pre-wrap">${response}</span></div>`;

    questionInput.value = ''; // Clear input
    chatbox.scrollTop = chatbox.scrollHeight; // Scroll to bottom
}


// --- Feature 4: Interview Scheduler ---
async function handleScheduling() {
    const recruiterSlots = document.getElementById('recruiter-slots').value;
    const candidateSlots = document.getElementById('candidate-slots').value;
    const outputDiv = document.getElementById('scheduler-output');

    if (!recruiterSlots || !candidateSlots) {
        displayError('scheduler-output', 'Please provide availability for both the recruiter and candidates.');
        return;
    }
    
    showLoader('scheduler-loader');
    outputDiv.innerHTML = '';

    const result = await eel.schedule_interviews(recruiterSlots, candidateSlots)();
    
    hideLoader('scheduler-loader');
    if (result.error) {
        displayError('scheduler-output', result.error);
    } else {
        renderScheduleResult(result);
    }
}

function renderScheduleResult(result) {
    const outputDiv = document.getElementById('scheduler-output');
    let scheduleHtml = '<p class="text-slate-500">No interviews could be scheduled.</p>';
    let unscheduledHtml = '';

    if (result.schedule && result.schedule.length > 0) {
        scheduleHtml = `
            <h3 class="text-lg font-semibold mb-2 text-green-700">Final Schedule</h3>
            <ul class="divide-y divide-slate-200">
                ${result.schedule.map(item => `
                    <li class="py-3 flex justify-between items-center">
                        <span class="font-medium text-slate-800">${item.candidate}</span>
                        <span class="text-sm text-slate-600">${item.slot}</span>
                    </li>
                `).join('')}
            </ul>
        `;
    }

    if (result.unscheduled && result.unscheduled.length > 0) {
        unscheduledHtml = `
            <h3 class="text-lg font-semibold mt-6 mb-2 text-amber-700">Unscheduled Candidates</h3>
            <ul class="divide-y divide-slate-200">
                ${result.unscheduled.map(item => `
                    <li class="py-3 flex justify-between items-center">
                        <span class="font-medium text-slate-800">${item.candidate}</span>
                        <span class="text-sm text-slate-500">${item.reason}</span>
                    </li>
                `).join('')}
            </ul>
        `;
    }

    outputDiv.innerHTML = scheduleHtml + unscheduledHtml;
}
