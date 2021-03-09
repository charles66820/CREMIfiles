<template>
  <section>
    <h2>Server images list component</h2>
    <ul class="imageList">
      <li v-for="image in images" :key="image.id">
        <span class="imageTitle">{{ image.name }}</span>
        <img :src="image.blob" alt="Images" />
      </li>
    </ul>
    <ul>
      <li v-for="err in errors" :key="err.type">
        {{ err.message }}
      </li>
    </ul>
  </section>
</template>

<script>
import httpApi from "../http-api.js";
export default {
  name: "Images",
  props: {
    msg: String,
  },
  data() {
    return {
      images: [],
      errors: [],
    };
  },
  methods: {},
  mounted: function () {
    httpApi
      .getImages()
      .then((res) => {
        this.images = res.data;
        for (let image of this.images) {
          httpApi.getImageBlob(image.id).then((res) => {
            let reader = new window.FileReader();
            reader.readAsDataURL(res.data);
            reader.addEventListener("load", () => (image.blob = reader.result));
          });
        }
      })
      .catch((err) => this.errors.push(err));
  },
};
</script>

<style scoped>
ul.imageList {
  list-style-type: none;
  display: flex;
  flex-wrap: wrap;
  margin: -4px;
}

.imageList li {
  position: relative;
  height: 40vh;
  margin: 4px;
}

.imageList li img {
  height: 100%;
}

.imageTitle {
  position: absolute;
  top: 8px;
  left: 8px;
  color: white;
  font-size: 1.5em;
  margin: 0;
  font-weight: bold;
  -webkit-text-stroke: 0.8px black;
}

@media (max-aspect-ratio: 1/1) {
  .imageList li {
    max-height: 30vh;
    overflow: auto;
  }
}

@media (max-height: 480px) {
  .imageList li {
    height: 80vh;
  }
}
</style>
