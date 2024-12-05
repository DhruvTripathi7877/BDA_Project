const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const app = express();
const PORT = process.env.PORT || 5000;
const bodyParser = require("body-parser");
const {JWT_SECRET} = require("./config");
const multer = require('multer');

const storage = multer.memoryStorage()

const upload = multer({ storage: storage })

const corsOptions = {
  "origin": "*",
  "methods": "GET,HEAD,PUT,PATCH,POST,DELETE",
  "preflightContinue": false,
  "optionsSuccessStatus": 204
}

app.use(cors(corsOptions));
const {indexRouter} = require("./routes/index");
const { authMiddleware } = require("./middlewares/middleware");
const { predictRouter } = require("./routes/predict");

app.use(bodyParser.json());

// RHa57kyDkETv41ap
const MONGO_URL = "mongodb+srv://dhruvtripathi7777:RHa57kyDkETv41ap@cluster0.iqwhz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongoose.connect(MONGO_URL)
    .then(() => {
        console.log('Connected To MongoDB')
    })
    .catch((err) => {
        console.log("MongoDB Connection Error")
    })

app.use("/",indexRouter)
app.use("/predict",upload.single('file'),predictRouter)

app.listen(PORT, () => {
    console.log(`Server started at PORT:${PORT}`)
});