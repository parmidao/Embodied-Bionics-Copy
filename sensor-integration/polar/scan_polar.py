import asyncio
from bleak import BleakScanner

async def main():
    print("Scanning 10s... (strap must be worn + moist)")
    devices = await BleakScanner.discover(timeout=10.0)
    for d in devices:
        if d.name and "Polar" in d.name:
            print(f"FOUND  {d.address}  |  {d.name}")
    print("done")

asyncio.run(main())