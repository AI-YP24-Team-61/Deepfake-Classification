import asyncio
import aiohttp


API_URL = 'http://127.0.0.1:8000'


async def post_data(endpoint: str, input_data, timeout=0):
    # print(f'Начал выполнение: {idx}')
    url = f'{API_URL}/{endpoint}'
    print(url)
    await asyncio.sleep(timeout)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=input_data) as response:
            # if url == "http://127.0.0.1:8000/api/v1/models/predict":
            #     print(f"Predict is done in {time.time():.2f}")
            return await response.json()


async def get_data(endpoint: str):
    # print(f'Начал выполнение: {idx}')
    url = f'{API_URL}/{endpoint}'
    print(url)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
