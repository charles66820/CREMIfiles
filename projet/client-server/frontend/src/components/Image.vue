<template>
  <div>
    <h1>{{ msg }}</h1>

    <!-- @click= -->

    <select v-model="selected" @change="onImageChange($event)">
      <option disabled selected value>-- select an image --</option>
      <option v-for="image in images" :key="image.id" :value="image.id">
        {{ image.name }}
      </option>
    </select>

    <h3>Selected image id {{ selected }}</h3>
    <img v-if="selected != null" id="previewImage" alt="Images" />

    <ul>
      <li v-for="err in errors" :key="err.type">
        {{ err.message }}
      </li>
    </ul>

    <form v-on:submit="onSubmitImage($event)">
      <input type="file" ref="image" name="image" />
      <button type="submit">Submit</button>
    </form>
  </div>
</template>

<script>
import httpApi from "../http-api.js";
export default {
  name: "Image",
  props: {
    msg: String,
  },
  data() {
    return {
      images: [],
      errors: [],
      selected: null,
    };
  },
  methods: {
    onImageChange(e) {
      httpApi.getImageBlob(e.target.value).then((res) => {
        let reader = new window.FileReader();
        reader.readAsDataURL(res.data);
        reader.onload = () => {
          document
            .querySelector("#previewImage")
            .setAttribute("src", reader.result);
        };
      });
    },
    onSubmitImage(e) {
      // this.$refs.image.files[0] but i prefer the event way
      let file = e.target["image"].files[0];
      httpApi.postImage(file)
        .then((res) => {
          this.images.push({ id: res.data.id, name: file.name });
        })
        .catch((err) => this.errors.push(err));

      e.preventDefault();
    },
  },
  mounted: function () {
    httpApi.getImages()
      .then((res) => (this.images = res.data))
      .catch((err) => this.errors.push(err));
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
</style>
