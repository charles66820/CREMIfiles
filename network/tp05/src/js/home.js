let show = true;
function toggleList() {
    let elem = document.getElementById("listAMasque");
    if (show)
        elem.style.visibility = "hidden";
    else
        elem.style.visibility = "visible";
    show = !show;
}