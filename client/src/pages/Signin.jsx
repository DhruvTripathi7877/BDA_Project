import { useState } from "react"
import Button from "../components/Button"
import Footer from "../components/Footer"
import HeaderComponent from "../components/HeaderComponent"
import InputBox from "../components/InputBox"
import SubHeading from "../components/SubHeading"
import axios from "axios"
import { useNavigate } from "react-router-dom"

function Signin()
{
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");   

    const navigate = useNavigate();

    function signInUser()
    {
        axios.post("http://localhost:5000/user/signin",{
            username:email,
            password:password
        }).then((res)=>{
            localStorage.setItem("token",res.data.token)
            navigate("/dashboard")
        })
    }

    return (
        <div className="bg-black bg-opacity-70 w-screen h-screen flex justify-center items-center">
            <div className="bg-white flex flex-col rounded-md  px-4 w-[340px]">
                <HeaderComponent text="Sign In" />
                <SubHeading label="Enter your credentials to access your account"/>
                <InputBox text="Email" placeholder="johndoe@example.com" isPassword={false} onChange={e=>setEmail(e.target.value)}/>
                <InputBox text="Password" placeholder="" isPassword={true} onChange={e=>setPassword(e.target.value)}/>
                <Button text="Sign In" onClick={signInUser}/>
                <Footer text="Don't have an account?" text2="Sign Up" />
            </div>
        </div>
    )
}

export {Signin}