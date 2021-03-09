<template>
  <section>
    <h2>Image component for look server images form <code>select</code></h2>

    <ul>
      <li v-for="err in errors" :key="err.type">
        {{ err.message }}
      </li>
    </ul>

    <select v-model="selected" @change="onImageChange($event)">
      <option disabled selected value>-- select an image --</option>
      <option v-for="image in images" :key="image.id" :value="image.id">
        {{ image.name }}
      </option>
    </select>

    <p>Selected image id is {{ selected }}</p>
    <img v-if="selected != null" id="previewImage" alt="Images" />
  </section>
  <section>
    <h2>Upload new image</h2>
    <form v-on:submit="onSubmitImage($event)">
      <input type="file" ref="image" name="image" />
      <button type="submit">Submit</button>
    </form>
  </section>
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
      httpApi
        .postImage(file)
        .then((res) => {
          this.images.push({ id: res.data.id, name: file.name });
        })
        .catch((err) => this.errors.push(err));

      e.preventDefault();
    },
  },
  mounted: function () {
    httpApi
      .getImages()
      .then((res) => (this.images = res.data))
      .catch((err) => this.errors.push(err));
  },
};
</script>

<style scoped>
#previewImage {
  max-height: 50%;
  max-width: 50%;
}
</style>
