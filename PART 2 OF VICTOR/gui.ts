<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Next-Gen Modular Architecture Simulator</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
        .status-dot {
            height: 10px;
            width: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-running { background-color: #22c55e; }
        .status-degraded { background-color: #f59e0b; }
        .status-error { background-color: #ef4444; }
        .status-initializing { background-color: #3b82f6; }
        .log-entry {
            border-left: 3px solid;
            transition: all 0.3s ease;
        }
        .log-info { border-color: #3b82f6; }
        .log-warn { border-color: #f59e0b; }
        .log-error { border-color: #ef4444; }
        .log-success { border-color: #22c55e; }
    </style>
</head>
<body class="bg-gray-900 text-gray-200">

    <div id="app" class="container mx-auto p-4 md:p-8">
        <header class="text-center mb-8">
            <h1 class="text-4xl font-bold text-white mb-2">Next-Gen Architecture Dashboard</h1>
            <p class="text-lg text-gray-400">Simulating a Self-Healing, Parallel, and Modular System</p>
        </header>

        <div class="bg-gray-800 rounded-lg shadow-xl p-6 mb-8">
            <h2 class="text-2xl font-semibold text-white mb-4">System Controls</h2>
            <div class="flex flex-wrap gap-4">
                <input type="text" id="componentName" placeholder="New component name" class="bg-gray-700 text-white rounded-md px-4 py-2 flex-grow focus:outline-none focus:ring-2 focus:ring-indigo-500">
                <button id="addComponent" class="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-md transition-colors duration-300">
                    Add Component
                </button>
            </div>
        </div>

        <main id="component-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <!-- Components will be dynamically inserted here -->
        </main>
    </div>

    <script type="module">
        // --- State Management ---
        let components = [];
        let logHistory = {};

        // --- Core Functions ---

        /**
         * Adds a new component to the system.
         * @param {string} name - The name of the new component.
         */
        function addComponent(name) {
            if (!name || components.some(c => c.name === name)) {
                alert("Please enter a unique name for the component.");
                return;
            }
            const componentId = `comp_${Date.now()}`;
            const newComponent = {
                id: componentId,
                name,
                status: 'Initializing',
                version: 1,
                cpuUsage: 0,
                memoryUsage: 0,
                logs: [],
            };
            components.push(newComponent);
            logEvent(componentId, `Component '${name}' created and is initializing.`, 'info');
            render();
        }
        
        /**
         * Removes a component from the system.
         * @param {string} componentId - The ID of the component to remove.
         */
        function removeComponent(componentId) {
            components = components.filter(c => c.id !== componentId);
            delete logHistory[componentId];
            render();
        }

        /**
         * Simulates a hot-reload of a component.
         * @param {string} componentId - The ID of the component to hot-reload.
         */
        function hotReload(componentId) {
            const component = components.find(c => c.id === componentId);
            if (component) {
                component.version += 1;
                component.status = 'Initializing';
                logEvent(componentId, `Hot-reloading to v${component.version}. New code injected.`, 'success');
                setTimeout(() => {
                    component.status = 'Running';
                     logEvent(componentId, `Hot-reload complete. Now running v${component.version}.`, 'success');
                }, 2000);
                render();
            }
        }

        /**
         * Logs an event for a specific component.
         * @param {string} componentId - The ID of the component.
         * @param {string} message - The log message.
         * @param {string} type - The log type (info, warn, error, success).
         */
        function logEvent(componentId, message, type) {
            const component = components.find(c => c.id === componentId);
            if (component) {
                const log = {
                    message,
                    type,
                    timestamp: new Date().toLocaleTimeString()
                };
                component.logs.unshift(log);
                if (component.logs.length > 20) {
                    component.logs.pop();
                }
            }
        }


        // --- Simulation Engine ---

        /**
         * Simulates system behavior, including status changes and resource usage.
         */
        function simulateSystem() {
            components.forEach(component => {
                // Simulate CPU and Memory usage
                component.cpuUsage = Math.random() * 100;
                component.memoryUsage = Math.random() * 100;
                
                // Simulate status changes
                const randomEvent = Math.random();
                if (component.status !== 'Initializing') {
                    if (randomEvent < 0.02 && component.status !== 'Error') { // 2% chance of error
                        component.status = 'Error';
                        logEvent(component.id, 'Critical error detected! Attempting to self-heal.', 'error');
                        // Self-healing mechanism
                        setTimeout(() => {
                            if (components.some(c => c.id === component.id)) {
                                component.status = 'Running';
                                logEvent(component.id, 'Self-healing successful. Component restarted and is now stable.', 'success');
                            }
                        }, 5000);
                    } else if (randomEvent < 0.05 && component.status === 'Running') { // 5% chance of degraded performance
                        component.status = 'Degraded';
                        logEvent(component.id, 'Performance degraded. Anomalous input detected.', 'warn');
                    } else if (component.status === 'Degraded' && randomEvent > 0.6) { // 40% chance of recovery from degraded
                        component.status = 'Running';
                        logEvent(component.id, 'Performance has returned to normal.', 'info');
                    } else if (component.status === 'Initializing' && randomEvent > 0.5) {
                        component.status = 'Running';
                        logEvent(component.id, 'Initialization complete. Component is now running.', 'info');
                    }
                }
                 // Add random informational logs
                if(Math.random() < 0.1) {
                    logEvent(component.id, `Processing batch job #${Math.floor(Math.random()*1000)}.`, 'info');
                }
            });
            render();
        }


        // --- UI Rendering ---

        /**
         * Renders the entire application UI.
         */
        function render() {
            const grid = document.getElementById('component-grid');
            if (!grid) return;
            
            grid.innerHTML = components.map(component => `
                <div class="bg-gray-800 rounded-lg shadow-lg p-6 flex flex-col h-full" id="${component.id}">
                    <div class="flex-grow">
                        <div class="flex justify-between items-start mb-4">
                            <div>
                                <h3 class="text-xl font-bold text-white">${component.name}</h3>
                                <p class="text-sm text-gray-400">Version ${component.version}</p>
                            </div>
                            <div class="text-right">
                                <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-700 text-white">
                                    <span class="status-dot status-${component.status.toLowerCase()}"></span>
                                    ${component.status}
                                </span>
                            </div>
                        </div>

                        <div class="grid grid-cols-2 gap-4 mb-4 text-center">
                            <div class="bg-gray-700 p-2 rounded-md">
                                <p class="text-sm text-gray-400">CPU</p>
                                <p class="text-lg font-semibold text-white">${component.cpuUsage.toFixed(1)}%</p>
                            </div>
                             <div class="bg-gray-700 p-2 rounded-md">
                                <p class="text-sm text-gray-400">Memory</p>
                                <p class="text-lg font-semibold text-white">${component.memoryUsage.toFixed(1)}%</p>
                            </div>
                        </div>

                        <h4 class="text-md font-semibold text-gray-300 mb-2">Logs</h4>
                        <div class="bg-gray-900 rounded-md p-3 h-48 overflow-y-auto text-sm font-mono">
                            ${component.logs.map(log => `
                                <div class="log-entry log-${log.type} pl-3 mb-2">
                                    <span class="text-gray-500">${log.timestamp}</span> - ${log.message}
                                 </div>
                            `).join('') || '<p class="text-gray-500">No logs yet.</p>'}
                        </div>
                    </div>
                    <div class="mt-6 flex gap-4">
                        <button data-action="hot-reload" data-id="${component.id}" class="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-md transition-colors duration-300">
                            Hot-Reload
                        </button>
                        <button data-action="remove" data-id="${component.id}" class="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-md transition-colors duration-300">
                            Remove
                        </button>
                    </div>
                </div>
            `).join('');
        }

        // --- Event Listeners ---
        document.getElementById('addComponent').addEventListener('click', () => {
            const input = document.getElementById('componentName');
            addComponent(input.value);
            input.value = '';
        });

        document.getElementById('component-grid').addEventListener('click', (e) => {
            const target = e.target.closest('button');
            if (!target) return;
            
            const action = target.dataset.action;
            const id = target.dataset.id;

            if (action === 'hot-reload') {
                hotReload(id);
            } else if (action === 'remove') {
                removeComponent(id);
            }
        });


        // --- Initialization ---
        function init() {
            // Add some initial components to start
            addComponent('Auth Service');
            addComponent('Data Processor');
            addComponent('AI Inference Engine');

            // Start the simulation loop
            setInterval(simulateSystem, 2000);
        }

        init();
    </script>
</body>
</html>
