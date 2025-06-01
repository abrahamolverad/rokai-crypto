import asyncio
import websockets
import json

async def test_birdeye_ws():
    url = "wss://public-api.birdeye.so/socket/solana?x-api-key=234ae514274b477885a8200210a3576d"

    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({
            "type": "subscribe",
            "channel": "token_new"
        }))
        print("Subscribed to token_new")

        while True:
            msg = await ws.recv()
            print("📩 Message received:", msg)

asyncio.run(test_birdeye_ws())
