<template>
  <article>
    <h1>Home</h1>
    <Image msg="Test msg" />
    <Images />
  </article>
</template>

<script>
// @ is an alias to /src
import Image from "@/components/Image.vue";
import Images from "@/components/Images.vue";

export default {
  name: "Home",
  components: {
    Image,
    Images,
  },
};
</script>
