const express = require("express");
const zod=require("zod");
const {User} = require("../models/userSchema");
const bcrypt = require("bcrypt");

const saltRounds = 10;
const myPlaintextPassword = 's0/\/\P4$$w0rD';

const {JWT_SECRET} = require("../config");
const {authMiddleware} = require("../middlewares/middleware") 

const jwt = require("jsonwebtoken");

const router = express.Router();

const signupSchema = zod.object({
    username: zod.string().email(),
    password: zod.string(),
    firstName: zod.string(),
    lastName: zod.string(),
})

router.post("/signup", async (req, res) => {
    const body = req.body;

    const {success} = signupSchema.safeParse(body);

    if(!success){
        return res.status(400).json({message: "Invalid Request"})
    }

    const user = await User.findOne({username: body.username});

    if(user){
        return res.status(400).json({message: "User Already Exists"})
    }

    bcrypt.hash(req.body.password, saltRounds,async(err,hash)=>{
        if(err){
            return res.status(500).json({message: "Internal Server Error"})
        }

        const dbUser = await User.create({
            username:body.username,
            password:hash,
            firstName:body.firstName,
            lastName:body.lastName
        })

        const userId= dbUser._id;

        const token = jwt.sign({userId: dbUser._id}, JWT_SECRET);

        res.status(201).json({
            message:"User Created",
            token:token,
        })
    })
})

const signinBody = zod.object({
    username:zod.string().email(),
    password:zod.string(),
})

router.post("/signin",async(req,res)=>{
    const {success} = signinBody.safeParse(req.body);

    if(!success){
        return res.status(400).json({message: "Invalid Request"})
    }
    try
    {
        const user = await User.findOne({
            username:req.body.username,
        })

        if(user)
        {
            bcrypt.compare(req.body.password ,user.password,(err,result)=>{
                
                if(err){
                    return res.status(500).json({message: "Internal Server Error"})
                }
                if(result)
                {
                    const token = jwt.sign({userId:user._id},JWT_SECRET);

                    return res.json({
                        token
                    })
                }
                else{
                    console.log("Invalid Credentials")
                    return res.status(400).json({message: "Invalid Credentials"})
                }
            })
        }
    }
    catch(err){
        console.log("Invalid Credentials2")
        return res.status(400).json({message: "Invalid Credentials"})
    }
})

module.exports = {userRouter:router};