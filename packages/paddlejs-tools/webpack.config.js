const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const { CleanWebpackPlugin }  = require('clean-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin')

module.exports = {
    mode: 'development',
    entry: {
        model: './model/modelTest.js'
    },
    devtool: 'inline-source-map',
    devServer: {
        host: '0.0.0.0',
        port: 9000
    },
    plugins: [
        new CleanWebpackPlugin({
            dangerouslyAllowCleanPatternsOutsideProject: true
        }),
        new HtmlWebpackPlugin({
            filename: 'model.html',
            chunks: ['model'],
            template: './model/index.html'
        }),
        new CopyWebpackPlugin({patterns: [{
            from: path.resolve(__dirname, '../../../models/convertedModels')
        }]})
    ],
    resolve: {
        // Add ".ts" and ".tsx" as resolvable extensions.
        extensions: ['.ts', '.js'],
        alias: {
            '@paddlejs/paddlejs-core': path.resolve(__dirname, '../paddlejs-core/src/'),
            '@paddlejs/paddlejs-backend-webgl': path.resolve(__dirname, '../paddlejs-backend-webgl/src'),
            '@paddlejs/paddlejs-backend-cpu': path.resolve(__dirname, '../paddlejs-backend-cpu/src')
        }
    },
    module: {
        rules: [
            {
                test: /\.ts$/,
                loader: 'ts-loader',
                exclude: /node_modules/
            }
        ]
    },
    output: {
        filename: '[name].js',
        path: path.resolve(__dirname, 'dist')
    }
};
