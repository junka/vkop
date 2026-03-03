class VkopModelViewer {
    constructor() {
        this.network = null;
        this.nodes = new vis.DataSet([]);
        this.edges = new vis.DataSet([]);
        this.modelData = null;
        
        this.init();
    }

    init() {
        const container = document.getElementById('network');
        
        container.style.width = '100%';
        container.style.height = '100%';
        const data = {
            nodes: this.nodes,
            edges: this.edges
        };
        
        const options = {
            nodes: {
                shape: 'box',
                font: {
                    size: 14,
                    face: 'Arial',
                    color: '#ffffff'
                },
                scaling: {
                    label: {
                        enabled: true,
                        min: 14,
                        max: 20
                    }
                },
                margin: 10,
                widthConstraint: {
                    maximum: 200
                }
            },
            edges: {
                arrows: 'to',
                smooth: {
                    type: 'continuous'
                },
                color: {
                    color: '#848484',
                    highlight: '#848484',
                    hover: '#848484'
                },
                width: 2
            },
            physics: {
                enabled: false,
                hierarchicalRepulsion: {
                    nodeDistance: 120
                }
            },
            interaction: {
                dragNodes: true,
                dragView: true,
                hideEdgesOnDrag: false,
                hideNodesOnDrag: false,
                hover: true,
                hoverConnectedEdges: true,
                keyboard: {
                    enabled: false,
                    speed: { x: 10, y: 10, zoom: 0.02 },
                    bindToWindow: true
                },
                multiselect: false,
                navigationButtons: false,
                selectable: true,
                selectConnectedEdges: true,
                tooltipDelay: 300,
                zoomView: true
            },
            layout: {
                improvedLayout: true,
                hierarchical: {
                    enabled: true,
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 150,
                    nodeSpacing: 100,
                    edgeMinimization: true,
                    blockShifting: true,
                    parentCentralization: true
                }
            },
            autoResize: true,
            clickToUse: false
        };

        this.network = new vis.Network(container, data, options);
        window.addEventListener('resize', () => {
            if (this.network) {
                this.network.setSize('100%', '100%');
                setTimeout(() => {
                    this.network.fit();
                }, 100);
            }
        });
        this.bindEvents();
        this.sendMessage({ command: 'loadModel' });
    }
    
    bindEvents() {
        this.network.on('click', (params) => {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                this.showNodeDetails(nodeId);
            }
        });
        
        document.getElementById('resetViewBtn').addEventListener('click', () => {
            this.network.fit();
        });
        
        document.getElementById('loadModelBtn').addEventListener('click', () => {
            this.sendMessage({ command: 'loadModel' });
        });
        
        window.addEventListener('message', (event) => {
            const message = event.data;
            switch (message.command) {
                case 'modelLoaded':
                    this.loadModel(message.data);
                    break;
                case 'loadError':
                    this.showError(message.error);
                    break;
            }
        });
    }
    
    sendMessage(message) {
        if (typeof acquireVsCodeApi !== 'undefined') {
            const vscode = acquireVsCodeApi();
            vscode.postMessage(message);
        }
    }
    
    loadModel(data) {
        this.modelData = data;
        this.buildGraph();
        document.getElementById('status').textContent = 'Model loaded successfully';
    }
    
    buildGraph() {
        const graphNodes = [];
        const graphEdges = [];
        
        Object.entries(this.modelData.nodes).forEach(([nodeName, node]) => {
            const color = this.getNodeColor(node.op_type);
            graphNodes.push({
                id: nodeName,
                label: `${nodeName}\n${node.op_type}`,
                title: this.getNodeTooltip(node),
                color: color,
                shape: 'box'
            });
            
            node.dependencies.forEach(dep => {
                graphEdges.push({
                    from: dep,
                    to: nodeName,
                    arrows: 'to'
                });
            });
        });

        this.modelData.inputs.forEach(input => {
            const inputId = `input_${input.name}`;
            graphNodes.push({
                id: `input_${input.name}`,
                label: `Input\n${input.name}`,
                title: `Shape: [${input.shape.join(', ')}]`,
                color: '#4CAF50',
                shape: 'ellipse'
            });
            Object.values(this.modelData.nodes).forEach(node => {
                if (node.inputs.some(i => i.name === input.name)) {
                    graphEdges.push({
                        from: inputId,
                        to: node.name,
                        arrows: 'to'
                    });
                }
            });
        });
        
        this.modelData.outputs.forEach(output => {
            const outputId = `output_${output.name}`;
            graphNodes.push({
                id: `output_${output.name}`,
                label: `Output\n${output.name}`,
                title: `Shape: [${output.shape.join(', ')}]`,
                color: '#F44336',
                shape: 'ellipse'
            });
            Object.values(this.modelData.nodes).forEach(node => {
                if (node.outputs.some(o => o.name === output.name)) {
                    graphEdges.push({
                        from: node.name,
                        to: outputId,
                        arrows: 'to'
                    });
                }
            });
        });
        
        this.nodes.clear();
        this.edges.clear();
        this.nodes.add(graphNodes);
        this.edges.add(graphEdges);
        
        setTimeout(() => {
            this.network.fit();
        }, 100);
    }
    
    getNodeColor(opType) {
        const colorMap = {
            'Conv': '#2196F3',
            'Gemm': '#FF9800',
            'MatMul': '#9C27B0',
            'Relu': '#4CAF50',
            'Add': '#FF5722',
            'Mul': '#795548',
            'Concat': '#607D8B',
            'Softmax': '#E91E63',
            'BatchNormalization': '#00BCD4'
        };
        return colorMap[opType] || '#9E9E9E';
    }
    
    getNodeTooltip(node) {
        let tooltip = `<b>${node.name}</b> (${node.op_type})\n`;
        tooltip += `\nInputs: ${node.inputs.map(i => `${i.name}[${i.shape.join(',')}]`).join(', ')}`;
        tooltip += `\nOutputs: ${node.outputs.map(o => `${o.name}[${o.shape.join(',')}]`).join(', ')}`;
        
        if (Object.keys(node.attributes).length > 0) {
            tooltip += '\n\nAttributes:';
            Object.entries(node.attributes).forEach(([key, value]) => {
                tooltip += `\n  ${key}: ${value}`;
            });
        }
        
        return tooltip;
    }
    
    showNodeDetails(nodeId) {
        const nodeInfo = document.getElementById('nodeInfo');
        
        if (nodeId.startsWith('input_')) {
            const inputName = nodeId.substring(6);
            const input = this.modelData.inputs.find(i => i.name === inputName);
            nodeInfo.innerHTML = `
                <h4>Input: ${input.name}</h4>
                <p><strong>Shape:</strong> [${input.shape.join(', ')}]</p>
            `;
        } else if (nodeId.startsWith('output_')) {
            const outputName = nodeId.substring(7);
            const output = this.modelData.outputs.find(o => o.name === outputName);
            nodeInfo.innerHTML = `
                <h4>Output: ${output.name}</h4>
                <p><strong>Shape:</strong> [${output.shape.join(', ')}]</p>
            `;
        } else {
            const node = this.modelData.nodes[nodeId];
            let html = `
                <h4>${node.name} (${node.op_type})</h4>
                <p><strong>Inputs:</strong></p>
                <ul>
                    ${node.inputs.map(i => `<li>${i.name} [${i.shape.join(', ')}]</li>`).join('')}
                </ul>
                <p><strong>Outputs:</strong></p>
                <ul>
                    ${node.outputs.map(o => `<li>${o.name} [${o.shape.join(', ')}]</li>`).join('')}
                </ul>
            `;
            
            if (Object.keys(node.attributes).length > 0) {
                html += '<p><strong>Attributes:</strong></p><ul>';
                Object.entries(node.attributes).forEach(([key, value]) => {
                    html += `<li><strong>${key}:</strong> ${value}</li>`;
                });
                html += '</ul>';
            }
            
            nodeInfo.innerHTML = html;
        }
    }
    
    showError(error) {
        document.getElementById('status').textContent = `Error: ${error}`;
        console.error('Load error:', error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new VkopModelViewer();
});