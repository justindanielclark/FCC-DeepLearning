const arr = [1, 2, 3, 4, 5, 6, 7, 8];
let iteration = 0;
const reduced = arr.reduce((acc, curr) => {
  console.log(`Iteration Number: ${iteration}`);
  iteration++;
  console.log(`acc: ${acc}`);
  console.log(`curr: ${curr}`);
  const new_acc = acc + curr;
  console.log(`new_acc: ${new_acc}`);
  console.log("=============");
  return new_acc;
});

console.log("^^^^^^^^^^^^^^^", `Reduced: ${reduced}`, "^^^^^^^^^^^^^^^");
