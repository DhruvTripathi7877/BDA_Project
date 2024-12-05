function Button({text, onClick}){
    return(
        <div className="flex items-center justify-center mb-4">
            <button className="w-[100%] h-10 bg-sky-500 rounded-lg mt-5 text-white" onClick={onClick}>{text}</button>
        </div>
    )
}

export default Button