var path = require('path')
const { VueLoaderPlugin } = require('vue-loader');
var webpack = require('webpack')

const mode = process.env.NODE_ENV || 'development';

module.exports = {
  mode,
  entry: './src/main.js',
  output: {
    path: path.resolve(__dirname, './build'),
    publicPath: '/build/',
    filename: '[name].js'
  },
  module: {
    rules: [
      { test: /\.vue$/, loader: 'vue-loader', options: { loaders: { } } },
      { test: /\.js$/, loader: 'babel-loader', exclude: /node_modules/ },
      { test: /\.css$/, loader: ['style-loader', 'css-loader'] },
      { test: /.(ttf|otf|eot|svg|woff(2)?)(\?[a-z0-9]+)?$/, use: [{ loader: 'file-loader', options: { name: '[name].[ext]', publicPath: '/build/' } }] },
      { test: /\.(png|jpg|gif|svg)$/, loader: 'file-loader', options: { name: '[name].[ext]?[hash]' } }
    ]
  },
  resolve: {
    alias: {
        'vue$': 'vue/dist/vue.esm.js',
        '@': path.resolve(__dirname, 'src')
        }
    },
  plugins: [
    new VueLoaderPlugin()
  ],
  devServer: {
    historyApiFallback: true,
    noInfo: true,
    contentBase: [
        path.resolve(__dirname, '.'),       // serves /index.html
        path.resolve(__dirname, './build')  // serves /build/fontawesome.js etc.
    ],
    watchContentBase: false,
    publicPath: '/build/',

    proxy: {
      "/api": {
        target: "http://localhost:5000",
        changeOrigin: true,
        secure: false,
      },
    }
  },
  performance: {
    hints: false
  },
  devtool: '#eval-source-map'
}

if (process.env.NODE_ENV === 'production') {
  module.exports.watch = false
  module.exports.devtool = '#source-map'
  // http://vue-loader.vuejs.org/en/workflow/production.html
  module.exports.plugins = (module.exports.plugins || []).concat([
    new webpack.DefinePlugin({
      'process.env': {
        NODE_ENV: '"production"'
      }
    }),
    new webpack.optimize.UglifyJsPlugin({
      sourceMap: true,
      compress: {
        warnings: false
      }
    }),
    new webpack.LoaderOptionsPlugin({
      minimize: true
    })
  ])
} else {
  module.exports.plugins = (module.exports.plugins || []).concat([
    new webpack.DefinePlugin({
      'process.env': {
        NODE_ENV: '"development"'
      }
    })
  ])
}
