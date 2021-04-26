const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const fs = require('fs');
const Detect = require('./precisionDetect');
// 设置服务器端口
const port = 8912;

app.listen(port);
console.log('Server started! At http://localhost:' + port);

const detect = new Detect('mobile');

//配置body-parser
// app.use(bodyParser.urlencoded({ extended: false }));
// app.use(bodyParser.json());
// app.use(express.json({limit: '500mb'}));
// app.use(express.urlencoded({limit: '500mb'}));

app.use(bodyParser.json({limit: '50mb'}));
app.use(bodyParser.urlencoded({limit: '50mb', extended: false}));


app.post('/save', function (req, res) {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'X-Requested-With');
    res.header('Access-Control-Allow-Methods', 'PUT, POST, GET, DELETE, OPTIONS');
    res.header('X-Powered-By', '3.2.1');
    res.header('Content-Type', 'application/json;charset=utf-8');
    console.log('收到请求');

    //req.query只能拿到get参数
    //post请求使用 body-parser拿到
    const data = req.body;
    if (!data) {
        res.send('fail');
    }

    const {fileName, result, ua, layerName, shape} = data;
    console.log(layerName, 'request layername')
    console.log(shape, 'request shape')
    detect.setPeerData(JSON.parse(result));
    const layerShape = JSON.parse(shape);

    // const arr = JSON.parse(data.result)
    // fs.writeFile(`../${fileName}.txt`, result, (err) => {
    //     if (err) {
    //         return console.error(err);
    //     }
    //     console.log("数据写入成功！");
    //     console.log("--------我是分割线-------------")
    //     console.log("读取写入的数据！");
    // });
    detect.predictCb = (nextLayerName, shape) => {
        console.log(nextLayerName, 'predictCb')
        res.send({status: 0, layerName: nextLayerName, shape});
    }
    detect.page.evaluate((layerName, shape) => {
        console.log('evaluate!!!!!!!!!!!!')
        window.layerName = layerName;
        window.paddlejs.modelConfig.fetchShape = shape || [];
        console.log(layerName, 'layerName before preheat')
        console.log(shape, 'shape before preheat')
        window.paddlejs.preheat();
    }, layerName, layerShape);

});
