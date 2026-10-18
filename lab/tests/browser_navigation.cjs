/* Optional browser regression: run against a local app or exported site. */
const {chromium}=require('playwright-core');
const assert=require('node:assert/strict');

(async()=>{
 const browser=await chromium.launch({headless:true,
   ...(process.env.BROWSER_EXECUTABLE?{executablePath:process.env.BROWSER_EXECUTABLE}:{}),
   args:process.env.BROWSER_NO_SANDBOX==='1'?['--no-sandbox']:[]});
 try{
  const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const url=process.env.DELTASCOPE_URL||'http://127.0.0.1:8890/';
  await page.goto(url);await page.waitForSelector('.image-card');
  assert.equal(await page.locator('#case-select').inputValue(),'beach');
  const initial=page.url(),length=await page.evaluate(()=>history.length);
  await page.locator('[data-tab="compare"]').click();
  assert.equal(await page.evaluate(()=>history.length),length,'Re-selecting the current view must not add history');
  await page.locator('[data-tab="lessons"]').click();const lessons=page.url();
  await page.locator('[data-case="space_station"]').click();const space=page.url();
  await page.locator('[data-strength="1"]').selectOption('1.5');
  await page.goBack();await page.waitForFunction(()=>document.querySelector('[data-strength="1"]').value==='1.25');
  assert.equal(page.url(),space);
  await page.goBack();await page.waitForFunction(()=>!document.querySelector('#lessons').hidden);
  assert.equal(page.url(),lessons);
  await page.goForward();await page.waitForFunction(()=>!document.querySelector('#compare').hidden);
  assert.equal(await page.locator('#case-select').inputValue(),'space_station');
  await page.goBack();await page.goBack();await page.waitForFunction(()=>!document.querySelector('#compare').hidden);
  assert.equal(page.url(),initial);assert.equal(await page.locator('#case-select').inputValue(),'beach');
  await page.locator('#seed-select').selectOption('171017');await page.goBack();
  await page.waitForFunction(()=>document.querySelector('#seed-select').value==='171018');
  await page.locator('#add-column').click();assert.equal(await page.locator('.image-card').count(),10);
  await page.goBack();await page.waitForFunction(()=>document.querySelectorAll('.image-card').length===8);
  await page.goForward();await page.waitForFunction(()=>document.querySelectorAll('.image-card').length===10);
  await page.goto(url+'#tab=guide');await page.waitForFunction(()=>!document.querySelector('#guide').hidden);
  await page.locator('[data-tab="compare"]').click();
  assert.equal(await page.locator('#case-select').inputValue(),'beach');assert.equal(await page.locator('.image-card').count(),8);
  await page.goBack();await page.waitForFunction(()=>!document.querySelector('#guide').hidden);
  await page.goForward();await page.waitForFunction(()=>!document.querySelector('#compare').hidden);
  assert.deepEqual(errors,[]);console.log('PASS: Back/Forward restores tabs, cases, strength, seed and columns; partial URLs reset to defaults.');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
