<template>
  <div class="home">
    <img alt="Vue logo" src="../assets/logo.png">
    <Image msg="Welcome to Your Vue.js App"/>
    <Images/>
  </div>
</template>

<script>
// @ is an alias to /src
import Image from '@/components/Image.vue'
import Images from '@/components/Images.vue'

export default {
  name: 'Home',
  components: {
    Image,
    Images,
  }
}
</script>
