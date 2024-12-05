import axios from "axios"
import Button from "../components/Button"
import { useState } from "react"

import {PropagateLoader} from 'react-spinners'
import FileUploader from "../components/FileUploader";


function Dashboard() {
  // const [isLoading, setIsLoading] = useState(false);
  // const [isData, setIsData] = useState(false)
  // const [disorderType, setDisorderType] = useState('')

  // const [userInput,setUserInput] = useState('')
  // function predictDisorder()
  // {
  //   setIsLoading(true)
  //   axios.post("http://localhost:5000/predict",{
  //     userInput
  //   }).then((res)=>{
  //     console.log(res.data)

  //     setIsLoading(false)
  //     setIsData(true)
  //     setDisorderType(res.data.output)
  //   })
  // }
  /*
  <PropagateLoader />
  */
  return (
    <div className="w-screen h-screen bg-black bg-opacity-70 flex flex-col justify-center items-center relative">
      <div className="absolute top-0">
        <div className=" text-white text-3xl font-semibold pt-10">
          Mental Disorder Predictor
        </div>
      </div>
      <div className="w-[350px]">
        <FileUploader/>
      </div>
      {/* <input onChange={(e)=>setUserInput(e.target.value)} className="bg-white h-[60px] w-[300px] -mt-40 border-2 border-slate-200 rounded-md font-semibold focus:outline-none focus:border-black px-2" placeholder="Enter Patient's Details"/>
      <div className="w-[300px]">
        <Button text="Analyse" onClick={predictDisorder}/>
      </div>
      {isLoading?<PropagateLoader className="mt-20"/>:null}
      <div className="text-white font-semibold text-lg pt-10">
        {isData?disorderType:null}
      </div> */}
    </div>
  )
}

export {Dashboard}