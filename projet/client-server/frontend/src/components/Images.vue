<template>
  <div>
    <div v-for="image in images" :key="image.id">
      {{ image.name }}
      <img :src="image.blob" alt="Images" />
    </div>

    <ul>
      <li v-for="err in errors" :key="err.type">
        {{ err.message }}
      </li>
    </ul>
  </div>
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
    httpApi.getImages()
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

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
</style>
