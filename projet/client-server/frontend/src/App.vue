<template>
  <header>
    <nav>
      <router-link class="nav-item icon" to="/">
        <img src="./assets/logo.png" alt="site icon" height="60">
      </router-link>
      <router-link class="nav-item active" to="/">Home</router-link>
      <router-link class="nav-item" to="/about">About</router-link>
    </nav>
  </header>
  <div id="nav">
  </div>
  <router-view/>
</template>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}
</style>
