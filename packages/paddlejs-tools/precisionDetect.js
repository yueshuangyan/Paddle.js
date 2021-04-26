const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const child_process = require('child_process');
const mode = 'python';
const {modelName, shape, w, h} = require('./modelConfig.json');
const precision = 0.02;

module.exports = class Dectect {
    constructor(mode) {
        this.mode = mode || '';
        this.peerData = [];
        this.page = null;
        this.predictCb = null;

        this.init();
    }

    setPeerData(data) {
        this.peerData = data;
    }

    convert(a, b, layer, shape, curLayers) {
        if (!a || !b) {
            console.log(`数据缺失: pc: ${!!a}, peer: ${!!b}`);
            return false;
        }
        if (a.length !== b.length) {
            console.log(`pc paddle: ${a.length}, peer: ${b.length} 长度不等`);
            return false;
        }

        let realPrecision = 0;
        let isAlign = true;

        for (let i = 0, len = a.length; i < len; i++) {
            if (isAlign && Math.abs(a[i] - b[i]) > precision) {
                console.log(`${layer}精度不对齐，从${i}开始，pc paddle: ${a[i]}; peer: ${b[i]}`);
                isAlign = false;
            }

            realPrecision = Math.max(realPrecision, Math.abs(a[i] - b[i]));
        }

        console.log(`${curLayers} ${layer} 精度${isAlign ? '': '未'}对齐, 实际精度误差： ${realPrecision}`);
        return isAlign;
    }

    runMobilePredict(detail, curLayerName, curShape, curLayers, nextLayerName, nextLayerShape) {
        const isAlign = this.convert(detail, this.peerData, curLayerName, curShape, curLayers);

        if (this.predictCb && (isAlign || !curLayerName)) {
            this.predictCb(nextLayerName, nextLayerShape);
            return;
        }
        else {
            console.log(`精度未对齐， 当前layerNum: ${curLayers - 1}`);
            // browser.close();
        }
    }

    runFluidPredict(layerName, callback) {
        console.log(layerName, 'runFluidPredict')
        console.log(modelName, 'modelName')
        console.log(shape, 'shape')
        const pythonProcess = child_process.spawn('python', [
            'run.py',
            '--model', modelName,
            '--opName', layerName,
            '--inputWidth', w,
            '--inputHeight', h,
            '--modelFileName', '__model__',
            '--paramFileName', 'params'
        ]);

        const chunks = [];

        pythonProcess.stdout.on('data', chunk => chunks.push(chunk));

        pythonProcess.stdout.on('end', () => {

            // If JSON handle the data
            const data = JSON.parse(Buffer.concat(chunks).toString());
            // console.log(data)

            callback(data);
        });

        pythonProcess.stderr.on('data', (data) => {
            console.log('error')
            console.error(`stderr: ${data}`);
        });


    }

    async init() {
        const browser = await puppeteer.launch({headless: false}); // headless必须设为false, 否则无响应
        const page = this.page = await browser.newPage();
        const me = this;

        // Expose a handler to the page
        await page.exposeFunction('onCustomEvent', ({detail, curLayerName, curShape, curLayers, nextLayerName, nextLayerShape}) => {
            // 在node脚本中执行

            // console.log(`Event fired: ${curLayerName}, detail: ${detail.slice(0, 10)}`);
            const align = function (data) {
                const isAlign = me.convert(detail, data, curLayerName, curShape, curLayers);

                if (isAlign && mode === 'python') {
                    if (mode === 'python') {
                        nextLayerName && page.evaluate(async () => {
                            window.layerName = window.nextLayerName;
                            const preheatRes = await window.paddlejs.preheat();
                            const event = new CustomEvent('preheat', { detail: preheatRes});
                            window.dispatchEvent(event);
                        })
                    }
                 }
                else {
                    console.log(`当前layerNum: ${curLayers - 1}`);
                    // browser.close();
                }
            }

            mode === 'python'
                ? this.runFluidPredict(curLayerName, align)
                : this.runMobilePredict(detail, curLayerName, curShape, curLayers, nextLayerName, nextLayerShape);
        });

        // listen for events of type 'status' and
        // pass 'type' and 'detail' attributes to our exposed function
        await page.evaluateOnNewDocument((mode) => {
            // 在page中执行
            console.log('evaluateOnNewDocument===============')
            let curLayerName = '';
            let curShape = [];
            let weightMap;
            const skipOps = ['split', 'concat', 'concat_mul']; // 多输出op跳过
            // const skipOps = []; // 多输出op跳过

            function getLayerByName(layerName) {
                if (!layerName) {
                    return;
                }

                for (let [index, item] of weightMap.entries()) {
                    if (item.opData && item.opData.outputTensors && item.opData.outputTensors[0] && item.opData.outputTensors[0].tensorId === layerName) {
                        window.curLayers = index;
                        return item.opData.outputTensors[0];
                    }
                }

                return false;
            }

            function setLayerId() {
                const layerName = window.layerName;
                for (let [index, item] of weightMap.entries()) {
                    if (item.opData && item.opData.outputTensors && item.opData.outputTensors[0] && item.opData.outputTensors[0].tensorId === layerName) {
                        window.curLayers = item.opData.iLayer;
                    }
                }
            }

            function getNextLayer() {
                const executorData = weightMap[++window.curLayers].opData;
                if (!executorData) {
                    return;
                }
                // op跳过
                if (skipOps.includes(executorData.name) || (executorData.outputTensors && executorData.outputTensors[0] && executorData.outputTensors[0].tensorId.endsWith('_packed'))) {
                    return getNextLayer();
                }
                else {
                    return executorData && executorData.outputTensors && executorData.outputTensors[0];
                }
            }

            async function predict(output, detail, curLayerName, curShape) {
                console.log('predict==========')
                console.log(curLayerName, 'curLayerName==========')
                // if ((output.tensorId && output.shape)) {
                if (output && output.tensorId && output.shape) {
                    window.layerName = output.tensorId;
                    window.paddlejs.modelConfig.fetchShape = output.shape;
                }


                // 当前的detail与下一层的layerName & shape
                if (!curLayerName) {
                    const preheatRes = await window.paddlejs.preheat();
                    const event = new CustomEvent('preheat', { detail: preheatRes});
                    window.dispatchEvent(event);
                }
                else {
                    // if (!curLayerName) {
                    //     const executorData = weightMap[window.curLayers - 1].opData;
                    //     const output = executorData && executorData.outputTensors && executorData.outputTensors[0] && executorData.outputTensors[0] || {};
                    //     curLayerName = output.tensorId;
                    //     curShape = output.shape;
                    // }
                    window.onCustomEvent({detail, curLayerName, curShape, curLayers: window.curLayers, nextLayerName: window.nextLayerName, nextLayerShape: window.nextLayerShape});
                }

                    // return true;
                // }

                // else {
                //     console.log(`前${window.curLayers}层已对齐，当前layerName: ${curLayerName}, shape: ${curShape}`);
                //     return false;
                // }
            }

            window.addEventListener('loaded', ({ type, detail}) => {
                window.curLayers = 0;
                weightMap = window.weightMap;
                window.layerName = '';
            });

            window.addEventListener('preheat', async ({type, detail}) => {
                console.log('preheat==========')
                // const {layerName, data} = detail;
                const data = detail;
                const layerName = window.layerName;

                if (!layerName) {
                    // 寻找第一个layer
                    window.curLayers = 0;
                    // output = getNextLayer() || {};
                    // return;
                }

                else {
                    // 获取当前curLayers值
                    setLayerId();
                }

                console.log(window.layerName, '=========layerName')
                console.log(window.curLayers, '=========curLayers')

                let output;
                if (window.curLayers === 0 && mode === 'python') {
                    output = getNextLayer();
                }
                else {
                    if (layerName) {
                        output = getLayerByName(layerName);
                    }
                    const nextLayerData = getNextLayer();
                    window.nextLayerName = nextLayerData?.tensorId;
                    window.nextLayerShape = nextLayerData?.shape;
                }

                const res = await predict(output, data, layerName, curShape);
                if (res) {
                    curShape = res.shape;
                }
                else {
                    // browser.close();
                }
            });
        }, mode);

        await page.goto('http://localhost:9000/model.html');

        return page;
        // await browser.close();
    };
}
