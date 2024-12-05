const {JWT_SECRET} = require("../config");
const jwt = require("jsonwebtoken");

const authMiddleware = async (req, res, next) => {
    // console.log("Auth Middleware")
    const authHeader = req.headers.authorization;
    if(!authHeader || !authHeader.startsWith("Bearer ")){
        return res.status(401).json({message: "Unauthorized"})
    }

    const token = authHeader.split(" ")[1];

    jwt.verify(token, JWT_SECRET,function(err, decoded){
        if(err){
            return res.status(401).json({message: "Unauthorized"})
        }
        req.userId = decoded.userId;
        next();
    });
}

module.exports = {authMiddleware};