1. Make sure you have the Demo Stack successfully running.

2. Ensure you are using the version of Node that comes with the
   platform-dependencies

3. Allocate budget to the hello account with:

```
curl http://localhost:9985/v1/accounts/hello/budget -d '{ "USD/1M":123456789000 }'
```

4. From the root of the rtbkit folder, type:

```
node nodeagents/nodebidagent.js
```
