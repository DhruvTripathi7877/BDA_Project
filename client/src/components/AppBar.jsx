
function AppBar()
{
    return(
        <div className="h-20 w-full border-b-2 border-slate-200 flex justify-between px-6 items-center shadow-sm">
            <div className="text-2xl font-bold">
                Payments App
            </div>
            <div className="flex items-center space-x-2">
                <div className=" font-semibold ">
                    Hello, User
                </div>
                <div className="w-10 h-10 text-black bg-black bg-opacity-15 rounded-full text-center">
                    <div className="pt-2">
                        D
                    </div>
                </div>
            </div>
        </div>
    )
}

export {AppBar}