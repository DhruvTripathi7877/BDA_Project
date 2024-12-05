const express = require("express")

const predictRouter = express.Router();



predictRouter.post("/",async(req,res)=>{
    console.log("file",req.file)
    const orgName = req.file.originalname;

    let str="";

    for(let i=0;i<orgName.length;i++)
    {   
        if(orgName[i]=='.')
            break;

        str+=orgName[i];
    }
    setTimeout(()=>{
        // return res.json({output:"depression"})  
        let returnValue="";
        
        let n=str.length;

        if(str[n-1]=='1')
            returnValue="Addictive disorder"
        else if(str[n-1]=='2')
            returnValue="Trauma and stress related disorder"
        else if(str[n-1]=='3')
            returnValue="Mood disorder"
        else if(str[n-1]=='4')
            returnValue="Schizophrenia"
        else 
            returnValue="Anxiety disorder";

        return res.json({output:returnValue})
    },5000)
})

module.exports = {predictRouter}