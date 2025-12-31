var path = require('path')
const VueLoaderPlugin = require('vue-loader/lib/plugin');
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
      { 
      test: /\.vue$/, 
      loader: 'vue-loader' 
      },
      { 
        test: /\.js$/, 
        loader: 'babel-loader', 
        exclude: /node_modules/ 
      },
      { 
        test: /\.css$/, 
        use: ['style-loader', 'css-loader'] 
      },
      { 
        // This rule handles Fonts AND SVGs now
        test: /.(ttf|otf|eot|svg|woff(2)?)(\?[a-z0-9]+)?$/, 
        use: [{ loader: 'file-loader', options: { name: '[name].[ext]', publicPath: '/build/' } }] 
      },
      { 
        // FIX: Removed '|svg' from this regex to prevent the conflict
        test: /\.(png|jpg|gif)$/, 
        loader: 'file-loader', 
        options: { name: '[name].[ext]?[hash]' } 
      }
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
    // 'static' replaces 'contentBase' in Webpack 5
    static: [
        { directory: path.resolve(__dirname, '.') },
        { directory: path.resolve(__dirname, './build') }
    ],
    devMiddleware: {
        publicPath: '/build/',
    },
    // FIX: Converted proxy to an Array format to satisfy the schema
    proxy: [
      {
        context: ['/api'],
        target: "http://localhost:5000",
        changeOrigin: true,
        secure: false,
      }
    ]
  },
  performance: {
    hints: false
  },
  // FIX 4: Removed the '#' prefix. Webpack 5 does not support legacy names.
  devtool: 'eval-source-map'
}

if (process.env.NODE_ENV === 'production') {
  module.exports.devtool = 'source-map' // FIX 4: Removed '#'
  
  module.exports.plugins = (module.exports.plugins || []).concat([
    new webpack.DefinePlugin({
      'process.env': {
        NODE_ENV: '"production"'
      }
    }),
    // FIX 5: Removed webpack.optimize.UglifyJsPlugin.
    // Webpack 5 minimizes automatically when mode is 'production'.
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