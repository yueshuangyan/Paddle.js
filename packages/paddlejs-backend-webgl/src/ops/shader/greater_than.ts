
/**
 * @file greater_than return x >= y
 */

function mainFunc(
    {},
    {}
) {
    return `
    // start函数
    void main(void) {
        ivec4 oPos = getOutputTensorPos();
        // 输出坐标转换为输入坐标
        float x = getValueFromTensorPos_input(oPos.r, oPos.g, oPos.b, oPos.a);
        float y = getValueFromTensorPos_counter(oPos.r, oPos.g, oPos.b, oPos.a);

        setOutput(bool(x >= y));
    }
    `;
}
export default {
    mainFunc,
    params: [
    ],
    textureFuncConf: {
        origin: ['getValueFromTensorPos'],
        counter: ['getValueFromTensorPos']
    },
    behaviors: [
    ],
    inputsName: [
        'X',
        'Y'
    ]
};
