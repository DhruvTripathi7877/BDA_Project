function InputBox({text, placeholder, onChange, isPassword})
{
    return (
        <div className="flex flex-col justify-center pt-4">
            <div className="font-semibold mb-2">
                {text}
            </div>
            <input type={(isPassword)?"password":"text"} placeholder={placeholder} className="border-2 border-slate-200 rounded-md w-full py-1 px-2 font-semibold" 
            onChange={onChange}/>
        </div>
    )
}

export default InputBox