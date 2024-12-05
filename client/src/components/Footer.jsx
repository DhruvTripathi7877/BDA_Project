import { Link } from "react-router-dom"

function Footer({text, text2})
{
    return (
        <div className="flex items-center justify-center mb-4 font-semibold text-zinc-800">
            <div className="pr-1">
                {text}
            </div>
            <Link to="/signin" className="underline">{text2}</Link>
        </div>
    )
}

export default Footer