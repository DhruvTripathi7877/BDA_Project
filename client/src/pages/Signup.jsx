// import 'HeaderComponent' from "../components/HeaderComponent";

import { Link, Navigate, useNavigate } from "react-router-dom"
import Button from "../components/Button"
import HeaderComponent from "../components/HeaderComponent"
import InputBox from "../components/InputBox"
import SubHeading from "../components/SubHeading"
import Footer from "../components/Footer"
import { useEffect, useState } from "react"
import axios from "axios"

function Signup()
{
    const [firstName, setFirstName] = useState("");
    const [lastName, setLastName] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");

    const navigate = useNavigate()

    function signUpUser()
    {
        axios.post("http://localhost:5000/user/signup",{
            username:email,
            password:password,
            firstName:firstName,
            lastName:lastName,
        }).then((res)=>{
            localStorage.setItem("token",res.data.token)
            navigate("/dashboard")
        })
    }

    return (
        <div className="bg-black bg-opacity-70 w-screen h-screen flex justify-center items-center">
            <div className="bg-white flex flex-col rounded-md  px-4 w-[340px]">
                <HeaderComponent text="Sign Up" />
                <SubHeading label="Enter your information to create an account"/>
                <InputBox text="First Name" placeholder="John" isPassword={false} onChange={(e)=>setFirstName(e.target.value)} />
                <InputBox text="Last Name" placeholder="Doe" isPassword={false} onChange={(e)=>setLastName(e.target.value)} />
                <InputBox text="Email" placeholder="johndoe@example.com" isPassword={false} onChange={e=>setEmail(e.target.value)} />
                <InputBox text="Password" placeholder="" isPassword={true} onChange={e=>setPassword(e.target.value)} />
                <Button text="Sign Up" onClick={signUpUser}/>
                <Footer text="Already have an account?" text2="Sign In" />
            </div>
        </div>
    )
}

export {Signup}