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
      // --- FIXED SECTION START ---
      { 
        // RULE 1: FONTS (Native Webpack 5)
        // Matches woff, woff2, ttf, eot, otf AND version strings like ?v=4.7.0
        test: /\.(woff|woff2|eot|ttf|otf)(\?.*)?$/,
        type: 'asset/resource',
        generator: {
          filename: '[name][ext]'
        }
      },
      { 
        // RULE 2: IMAGES & SVGs (Native Webpack 5)
        // Matches images and svgs
        test: /\.(png|jpg|gif|svg)(\?.*)?$/,
        type: 'asset/resource',
        generator: {
          filename: '[name][ext]'
        }
      }
      // --- FIXED SECTION END ---
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
    static: [
        { directory: path.resolve(__dirname, '.') },
        { directory: path.resolve(__dirname, './build') }
    ],
    devMiddleware: {
        publicPath: '/build/',
    },
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
  devtool: 'eval-source-map'
}

if (process.env.NODE_ENV === 'production') {
  module.exports.devtool = 'source-map'
  
  module.exports.plugins = (module.exports.plugins || []).concat([
    new webpack.DefinePlugin({
      'process.env': {
        NODE_ENV: '"production"'
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