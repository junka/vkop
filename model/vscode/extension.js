const vscode = require('vscode');
const path = require('path');
const fs = require('fs');

function activate(context) {
    console.log('VKOP Model Viewer activated');

    let disposable = vscode.commands.registerCommand('vkop.viewModel', async function (uri) {
        try {
            // 获取文件路径的多种方式
            let filePath;
            
            // 方式1: 从右键菜单传入的uri
            if (uri && uri.fsPath) {
                filePath = uri.fsPath;
            }
            // 方式2: 从活动编辑器获取
            else if (vscode.window.activeTextEditor) {
                filePath = vscode.window.activeTextEditor.document.fileName;
            }
            // 方式3: 让用户选择文件
            else {
                const options = {
                    canSelectMany: false,
                    openLabel: 'Select VKOP Model File',
                    filters: {
                        'VKOP Models': ['vkop', 'vkopbin']
                    }
                };
                
                const fileUri = await vscode.window.showOpenDialog(options);
                if (fileUri && fileUri[0]) {
                    filePath = fileUri[0].fsPath;
                }
            }
            
            // 验证文件
            if (!filePath) {
                vscode.window.showErrorMessage('No file selected');
                return;
            }
            
            const ext = path.extname(filePath).toLowerCase();
            if (ext !== '.vkop' && ext !== '.vkopbin') {
                vscode.window.showErrorMessage(`Please select a .vkop or .vkopbin file (selected: ${ext})`);
                return;
            }
            
            // 检查文件是否存在
            if (!fs.existsSync(filePath)) {
                vscode.window.showErrorMessage(`File not found: ${filePath}`);
                return;
            }

            console.log('Loading VKOP file:', filePath);

            // 创建Webview面板
            const panel = vscode.window.createWebviewPanel(
                'vkopViewer',
                `VKOP Model: ${path.basename(filePath)}`,
                vscode.ViewColumn.One,
                {
                    enableScripts: true,
                    retainContextWhenHidden: true
                }
            );

            panel.webview.html = getWebviewContent(context, panel.webview);

            // 处理来自webview的消息
            panel.webview.onDidReceiveMessage(
                async (message) => {
                    switch (message.command) {
                        case 'loadModel':
                            try {
                                const modelData = await parseVkopFile(filePath);
                                panel.webview.postMessage({
                                    command: 'modelLoaded',
                                    data: modelData
                                });
                            } catch (error) {
                                console.error('Parse error:', error);
                                panel.webview.postMessage({
                                    command: 'loadError',
                                    error: error.message
                                });
                            }
                            break;
                        case 'showMessage':
                            vscode.window.showInformationMessage(message.text);
                            break;
                    }
                },
                undefined,
                context.subscriptions
            );

        } catch (error) {
            console.error('Extension error:', error);
            vscode.window.showErrorMessage(`Failed to open VKOP viewer: ${error.message}`);
        }
    });

    context.subscriptions.push(disposable);
}

async function parseVkopFile(filePath) {
    console.log('Parsing file:', filePath);

    const buffer = await fs.promises.readFile(filePath);
    let offset = 0;

    function readString() {
        if (offset + 4 > buffer.length) {
            throw new Error('Unexpected end of file when reading string length');
        }
        const length = buffer.readUInt32LE(offset);
        offset += 4;

        if (offset + length > buffer.length) {
            throw new Error(`String length ${length} exceeds remaining buffer size`);
        }

        const str = buffer.toString('utf8', offset, offset + length);
        offset += length;
        return str;
    }
    function readDims() {
        if (offset + 4 > buffer.length) {
            throw new Error('Unexpected end of file when reading dims count');
        }
        const dimsCount = buffer.readUInt32LE(offset);
        offset += 4;
        const dims = [];
        for (let j = 0; j < dimsCount; j++) {
            if (offset + 4 > buffer.length) {
                throw new Error('Unexpected end of file when reading dim');
            }
            dims.push(buffer.readUInt32LE(offset));
            offset += 4;
        }
        return dims;
    }

    function readListWithShapes() {
        if (offset + 4 > buffer.length) {
            throw new Error('Unexpected end of file when reading list count');
        }
        const count = buffer.readUInt32LE(offset);
        offset += 4;
        const list = [];
        
        for (let i = 0; i < count; i++) {
            const name = readString();
            if (offset + 4 > buffer.length) {
                throw new Error('Unexpected end of file when reading shape count');
            }
            const shapeCount = buffer.readUInt32LE(offset);
            offset += 4;
            const shape = [];
            for (let j = 0; j < shapeCount; j++) {
                if (offset + 4 > buffer.length) {
                    throw new Error('Unexpected end of file when reading shape dimension');
                }
                shape.push(buffer.readUInt32LE(offset));
                offset += 4;
            }
            list.push({ name: name, shape: shape });
        }
        return list;
    }

    function readDict() {
        if (offset + 4 > buffer.length) {
            throw new Error('Unexpected end of file when reading dict count');
        }
        const count = buffer.readUInt32LE(offset);
        offset += 4;
        const dict = {};
        
        for (let i = 0; i < count; i++) {
            const key = readString();
            if (offset + 1 > buffer.length) {
                throw new Error('Unexpected end of file when reading value tag');
            }
            const tag = buffer.readUInt8(offset);
            offset += 1;
            
            let value;
            switch (tag) {
                case 0: // string
                    value = readString();
                    break;
                case 1: // int (64-bit)
                    if (offset + 8 > buffer.length) {
                        throw new Error('Unexpected end of file when reading int64 value');
                    }
                    const bigIntVal = buffer.readBigInt64LE(offset);
                    value = Number(bigIntVal);
                    offset += 8;
                    break;
                case 2: // float (32-bit)
                    if (offset + 4 > buffer.length) {
                        throw new Error('Unexpected end of file when reading float value');
                    }
                    value = buffer.readFloatLE(offset);
                    offset += 4;
                    break;
                case 3: // list of ints (uint32)
                    if (offset + 4 > buffer.length) {
                        throw new Error('Unexpected end of file when reading list count');
                    }
                    const intListCount = buffer.readUInt32LE(offset);
                    offset += 4;
                    const intList = [];
                    for (let j = 0; j < intListCount; j++) {
                        if (offset + 4 > buffer.length) {
                            throw new Error('Unexpected end of file when reading int list element');
                        }
                        intList.push(buffer.readUInt32LE(offset));
                        offset += 4;
                    }
                    value = intList;
                    break;
                case 4: // list of floats
                    if (offset + 4 > buffer.length) {
                        throw new Error('Unexpected end of file when reading float list count');
                    }
                    const floatListCount = buffer.readUInt32LE(offset);
                    offset += 4;
                    const floatList = [];
                    for (let j = 0; j < floatListCount; j++) {
                        if (offset + 4 > buffer.length) {
                            throw new Error('Unexpected end of file when reading float list element');
                        }
                        floatList.push(buffer.readFloatLE(offset));
                        offset += 4;
                    }
                    value = floatList;
                    break;
                case 5: // numpy array
                    // Read array metadata
                    const dataType = readString();
                    if (offset + 4 > buffer.length) {
                        throw new Error('Unexpected end of file when reading array dims count');
                    }
                    const dimsCount = buffer.readUInt32LE(offset);
                    offset += 4;
                    const dims = [];
                    for (let j = 0; j < dimsCount; j++) {
                        if (offset + 4 > buffer.length) {
                            throw new Error('Unexpected end of file when reading array dim');
                        }
                        dims.push(buffer.readUInt32LE(offset));
                        offset += 4;
                    }
                    // Read array data length
                    if (offset + 8 > buffer.length) {
                        throw new Error('Unexpected end of file when reading array data length');
                    }
                    const dataLength = Number(buffer.readBigUInt64LE(offset));
                    offset += 8;
                    
                    // Skip the actual data for now, just store metadata
                    if (offset + dataLength > buffer.length) {
                        throw new Error('Array data exceeds buffer size');
                    }
                    offset += dataLength;
                    
                    value = {
                        type: dataType,
                        shape: dims,
                        data_length: dataLength
                    };
                    break;
                default:
                    throw new Error(`Unknown value tag: ${tag}`);
            }
            dict[key] = value;
        }
        return dict;
    }

    try {
        // 解析模型数据
        console.log('Parsing inputs...');
        const inputs = readListWithShapes();
        console.log('Inputs:', inputs);
        
        console.log('Parsing outputs...');
        const outputs = readListWithShapes();
        console.log('Outputs:', outputs);
        
        if (offset + 4 > buffer.length) {
            throw new Error('Unexpected end of file when reading node count');
        }
        const nodeCount = buffer.readUInt32LE(offset);
        console.log('Node count:', nodeCount);
        offset += 4;
        
        const nodes = {};
        for (let i = 0; i < nodeCount; i++) {
            const opType = readString();
            const nodeName = readString();
            console.log(`Node ${i}: ${nodeName} (${opType})`);
            const attributes = readDict();
            const nodeInputs = readListWithShapes();
            const nodeOutputs = readListWithShapes();
            console.log('Node Inputs:', nodeInputs);
            console.log('Node Outputs:', nodeOutputs);
            
            if (offset + 4 > buffer.length) {
                throw new Error('Unexpected end of file when reading dependency count');
            }
            const depCount = buffer.readUInt32LE(offset);
            offset += 4;
            const dependencies = [];
            for (let j = 0; j < depCount; j++) {
                dependencies.push(readString());
            }
            
            if (offset + 4 > buffer.length) {
                throw new Error('Unexpected end of file when reading dependent count');
            }
            const depCount2 = buffer.readUInt32LE(offset);
            offset += 4;
            const dependents = [];
            for (let j = 0; j < depCount2; j++) {
                dependents.push(readString());
            }
            
            nodes[nodeName] = {
                op_type: opType,
                name: nodeName,
                attributes: attributes,
                inputs: nodeInputs,
                outputs: nodeOutputs,
                dependencies: dependencies,
                dependents: dependents
            };
        }
        
        if (offset + 4 > buffer.length) {
            throw new Error('Unexpected end of file when reading initializer count');
        }
        const initializerCount = buffer.readUInt32LE(offset);
        offset += 4;
        const initializers = {};
        const initializerOffsets = {};
        let totalMemorySize = 0;
        const alignment = 64;
        for (let i = 0; i < initializerCount; i++) {
            const name = readString();
            const dtype = readString();
            const dims = readDims(); 
            if (offset + 8 > buffer.length) {
                throw new Error('Unexpected end of file when reading data size');
            }
            const dataSize = Number(buffer.readBigUInt64LE(offset));
            offset += 8;
            if (offset + dataSize > buffer.length) {
                throw new Error('Initializer data exceeds buffer size');
            }
            const alignedOffset = (totalMemorySize + alignment - 1) & ~(alignment - 1);
            initializerOffsets[name] = alignedOffset;
            totalMemorySize = alignedOffset + dataSize;

            // let data;
            // data = buffer.slice(offset, offset + dataSize);
            
            initializers[name] = {
                dtype: dtype,
                dims: dims,
                data_size: dataSize,
                offset: alignedOffset,
                // data: data
            };
            offset += dataSize;
        }
        
        // 解析并发层级
        if (offset + 4 > buffer.length) {
            throw new Error('Unexpected end of file when reading level count');
        }
        const levelCount = buffer.readUInt32LE(offset);
        offset += 4;
        const concurrentLevels = [];
        for (let i = 0; i < levelCount; i++) {
            if (offset + 4 > buffer.length) {
                throw new Error('Unexpected end of file when reading level node count');
            }
            const nodeCountInLevel = buffer.readUInt32LE(offset);
            offset += 4;
            const level = [];
            for (let j = 0; j < nodeCountInLevel; j++) {
                level.push(readString());
            }
            concurrentLevels.push(level);
        }
        
        console.log('Parsing completed successfully');
        
        return {
            inputs: inputs,
            outputs: outputs,
            nodes: nodes,
            initializers: initializers,
            concurrent_levels: concurrentLevels
        };
    } catch (error) {
        console.error('Parsing failed:', error);
        throw error;
    }
}

function getWebviewContent(context, webview) {
    const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(context.extensionUri, 'media', 'main.js'));
    const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(context.extensionUri, 'media', 'styles.css'));
    const visUri = webview.asWebviewUri(vscode.Uri.joinPath(context.extensionUri, 'media', 'vis-network.min.js'));

    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VKOP Model Viewer</title>
    <link href="${styleUri}" rel="stylesheet">
    <script src="${visUri}"></script>
</head>
<body>
    <div class="container">
        <div class="toolbar">
            <button id="loadModelBtn">Load Model</button>
            <button id="resetViewBtn">Reset View</button>
            <span id="status">Ready</span>
        </div>
        <div class="content-area">
            <div id="network" class="network-container"></div>
            <div id="sidebar" class="sidebar">
                <h3>Node Details</h3>
                <div id="nodeInfo"></div>
            </div>
        </div>
    </div>
    <script src="${scriptUri}"></script>
</body>
</html>
    `;
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
};