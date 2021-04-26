import { Runner, env } from '@paddlejs/paddlejs-core';
import '@paddlejs/paddlejs-backend-webgl';
import modelConfig from '../modelConfig.json';
import { ops } from '@paddlejs/paddlejs-backend-webgl/ops';

const {modelName, fileCount, w, h} = modelConfig;
const modelPath = `http://localhost:9000/${modelName}/model.json`;

async function run() {
    const runner = new Runner({
        modelPath,
        feedShape: {
            fw: w,
            fh: h
        },
        fileCount,
        fill: '#fff',
        targetSize: { height: h, width: w }
    });
    window.paddlejs = runner;
    env.set('debug', true);

    // op校验
    await runner.load();
    const modelOps = runner.model.ops;
    const unsupportedOpList = new Set();
    const skipOpList = ['feed', 'fetch'];
    for (let op of modelOps) {
        if (!ops[op.type]) {
            if (skipOpList.includes(op.type)) {
                continue;
            }
            unsupportedOpList.add(op.type)
        }
    }

    if (unsupportedOpList.size) {
        console.log('以下算子不支持');
        console.log(unsupportedOpList)
    }

    await runner.init();
    window.weightMap = runner.weightMap;
    // window.layerName = 'fc_0.tmp_1';
    // window.layerName = 'batch_norm_3.tmp_2';

    let event = new CustomEvent('loaded', { detail: []});
    window.dispatchEvent(event);
    // window.layerName = "conv2d_0.tmp_0";
    // window.layerName = "BatchNormBackward20_bn.batch_norm.output.1.tmp_2";
    // window.layerName = "BatchNormBackward28_bn.batch_norm.output.1.tmp_2";
    const preheatRes = await runner.preheat();
    console.log(preheatRes)
    event = new CustomEvent('preheat', { detail: preheatRes});
    window.dispatchEvent(event);
    console.log(preheatRes);
}

run();
