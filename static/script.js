document.addEventListener("DOMContentLoaded", function () {
    const circles = document.querySelectorAll(".circle");
    const displayImage = document.getElementById("display-image");
    const displayTitle = document.getElementById("display-title");

    circles.forEach(circle => {
        circle.addEventListener("click", () => {
            const image = circle.getAttribute("data-image");
            const title = circle.getAttribute("data-title");

            displayImage.src = image;
            displayTitle.textContent = title;
        });
    });
});