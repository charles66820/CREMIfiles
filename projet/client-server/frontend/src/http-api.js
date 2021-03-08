import axios from "axios";

export default {
  getImages() {
    return axios.get("/images");
  },
  getImageBlob(id) {
    return axios
      .get("/images/" + id, { responseType: "blob" });
  },
  postImage(file) {
    let formData = new FormData();
    formData.append("file", file);
    return axios
      .post("/images", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
  }
}