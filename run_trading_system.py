import subprocess
import sys
import time

def main():
    # Set Python executable
    python_exec = sys.executable
    
    # Start processes
    try:
        # Start main trading bot
        trading_bot = subprocess.Popen(
            [python_exec, "binance_trading_bot.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Started trading bot process")
        
        # Start pump and dump analyzer
        pump_dump_analyzer = subprocess.Popen(
            [python_exec, "pump_dump_analyzer.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Started pump and dump analyzer process")
        
        # Start telegram monitor
        telegram_monitor = subprocess.Popen(
            [python_exec, "telegram_monitor.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Started telegram monitor process")
        
        # Keep main thread alive to monitor processes
        while True:
            # Check if processes are still running
            if trading_bot.poll() is not None:
                print("Trading bot process stopped, restarting...")
                trading_bot = subprocess.Popen(
                    [python_exec, "binance_trading_bot.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            if pump_dump_analyzer.poll() is not None:
                print("Pump and dump analyzer process stopped, restarting...")
                pump_dump_analyzer = subprocess.Popen(
                    [python_exec, "pump_dump_analyzer.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            if telegram_monitor.poll() is not None:
                print("Telegram monitor process stopped, restarting...")
                telegram_monitor = subprocess.Popen(
                    [python_exec, "telegram_monitor.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            time.sleep(5)  # Check every 5 seconds
    
    except KeyboardInterrupt:
        print("Stopping all processes...")
        trading_bot.terminate()
        pump_dump_analyzer.terminate()
        telegram_monitor.terminate()
        print("All processes stopped")

if __name__ == "__main__":
    main()
