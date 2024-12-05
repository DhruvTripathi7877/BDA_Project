import { useNavigate } from "react-router-dom";
import Button from "./Button";

function UserEntry({firstName,lastName,id})
{
    const navigate = useNavigate()
    function sendMoney(){
        navigate(`/send/?id=`+id+"&name="+`${firstName} ${lastName}`)
    }
    return (
        <div className="flex justify-between mx-5">
        <div className="flex items-center space-x-2">
            <div className="w-10 h-10 text-black bg-black bg-opacity-15 rounded-full text-center">
                <div className="pt-2 font-semibold">
                    {firstName[0]}{lastName[0]}
                </div>
            </div>
            <div className="text-xl font-semibold ">
                {firstName} {lastName}
            </div>
        </div>
        <div className="w-32">   
            <Button text="Send Money" onClick={sendMoney}/>
        </div>
      </div>
    )
}

export {UserEntry}