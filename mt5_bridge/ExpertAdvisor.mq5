//+------------------------------------------------------------------+
//|  ShermQuantyBridge.mq5                                           |
//|  Broker-side half of the Sherm Quanty execution bridge.          |
//+------------------------------------------------------------------+
//
//  STATUS: UNVERIFIED.
//
//  This file has never been compiled or run. It was written against the
//  MQL5 documentation and the protocol in protocol.py, but there is no
//  MetaTrader terminal in the development environment, so nothing here
//  has been executed even once. Treat it as a specification to review and
//  debug on your own terminal, not as working code.
//
//  The Python half (client.py) IS tested — 10/10 against fake_server.py,
//  which implements this same protocol. So the contract is pinned down
//  even though this implementation of it is not.
//
//  WHAT THIS MUST GUARANTEE
//
//  1. Idempotency. Every mutating request carries a request_id. If an id
//     has already been executed, replay the stored result and DO NOT act
//     again. The Python client retries aggressively on transport failure
//     and relies entirely on this to avoid double-opening a position.
//
//  2. Atomic stop attachment. A position is never opened without its stop
//     in the same order_send call. Never place-then-modify: the gap
//     between the two is where an unprotected position gets hit.
//
//  3. Honest failure. If the broker refuses, respond "rejected" with the
//     reason. Do not silently retry, and do not report success.
//
//  DEPLOYMENT
//    - Attach to any single chart. Symbol is irrelevant; commands name it.
//    - Tools > Options > Expert Advisors > Allow WebRequest / DLL imports
//      as required by the socket library you use.
//    - Set BridgePort to match config.MT_BRIDGE_PORT (default 9000).
//    - Bind to 127.0.0.1 only. This socket can move real money and has no
//      authentication; it must never be reachable off the host.
//
//  NOTE ON SOCKETS: MQL5's native socket API (SocketCreate etc.) is a
//  CLIENT api — it cannot listen for inbound connections. The options are
//  to invert the direction (EA dials out to a Python listener), or to use
//  a small DLL that provides a listening socket. Inverting is simpler and
//  avoids DLL permissions entirely; the protocol is unchanged either way
//  because it is request/response over a stream. Decide this before
//  debugging, and if you invert, Python runs FakeEAServer's role and the
//  EA runs the client's — the message shapes stay identical.
//
//+------------------------------------------------------------------+

#property copyright "Sherm Quanty"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

input int    BridgePort      = 9000;
input string BridgeHost      = "127.0.0.1";
input int    PollIntervalMs  = 100;
input int    MaxProcessedIds = 4096;   // idempotency ledger capacity

CTrade        trade;
CPositionInfo posinfo;

// --- Idempotency ledger --------------------------------------------------
// Parallel arrays: request_id -> serialised result JSON. A ring buffer is
// adequate; ids older than the buffer cannot still be in flight, since the
// Python client gives up after three attempts within a few seconds.
string g_processed_ids[];
string g_processed_results[];
int    g_processed_count = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   ArrayResize(g_processed_ids,     MaxProcessedIds);
   ArrayResize(g_processed_results, MaxProcessedIds);
   EventSetMillisecondTimer(PollIntervalMs);
   PrintFormat("[bridge] listening on %s:%d", BridgeHost, BridgePort);
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   Print("[bridge] stopped");
  }

void OnTimer()
  {
   string request = BridgeReceive();     // transport-specific; see note above
   if(StringLen(request) == 0)
      return;
   string response = HandleRequest(request);
   BridgeSend(response);
  }

//+------------------------------------------------------------------+
//| Idempotency                                                      |
//+------------------------------------------------------------------+
int FindProcessed(const string request_id)
  {
   for(int i = 0; i < g_processed_count && i < MaxProcessedIds; i++)
      if(g_processed_ids[i] == request_id)
         return(i);
   return(-1);
  }

void RecordProcessed(const string request_id, const string result_json)
  {
   int slot = g_processed_count % MaxProcessedIds;
   g_processed_ids[slot]     = request_id;
   g_processed_results[slot] = result_json;
   g_processed_count++;
  }

//+------------------------------------------------------------------+
//| Dispatch                                                         |
//+------------------------------------------------------------------+
string HandleRequest(const string raw)
  {
   string command    = JsonGetString(raw, "command");
   string request_id = JsonGetString(raw, "request_id");

   bool mutating = (command == "open" || command == "close"
                    || command == "modify_stop");

   if(mutating)
     {
      int prior = FindProcessed(request_id);
      if(prior >= 0)
        {
         // Replayed request. Return the ORIGINAL result; do not act again.
         PrintFormat("[bridge] duplicate %s id=%s — replaying", command, request_id);
         return(BuildResponse(request_id, "ok", g_processed_results[prior]));
        }
     }

   if(command == "ping")        return(BuildResponse(request_id, "ok", "\"pong\":true"));
   if(command == "account")     return(CmdAccount(request_id));
   if(command == "positions")   return(CmdPositions(request_id));
   if(command == "quote")       return(CmdQuote(request_id, JsonGetString(raw, "symbol")));
   if(command == "open")        return(CmdOpen(request_id, raw));
   if(command == "close")       return(CmdClose(request_id, raw));
   if(command == "modify_stop") return(CmdModifyStop(request_id, raw));

   return(BuildResponse(request_id, "error", "\"reason\":\"unknown command\""));
  }

//+------------------------------------------------------------------+
string CmdAccount(const string request_id)
  {
   string data = StringFormat(
      "\"balance\":%.2f,\"equity\":%.2f,\"margin\":%.2f,"
      "\"margin_free\":%.2f,\"currency\":\"%s\"",
      AccountInfoDouble(ACCOUNT_BALANCE),
      AccountInfoDouble(ACCOUNT_EQUITY),
      AccountInfoDouble(ACCOUNT_MARGIN),
      AccountInfoDouble(ACCOUNT_MARGIN_FREE),
      AccountInfoString(ACCOUNT_CURRENCY));
   return(BuildResponse(request_id, "ok", data));
  }

//+------------------------------------------------------------------+
//| Live positions. This is what Python treats as the source of      |
//| truth, so it must report stop_price faithfully — a position      |
//| showing no stop is how reconciliation detects an unprotected     |
//| position.                                                        |
//+------------------------------------------------------------------+
string CmdPositions(const string request_id)
  {
   string items = "";
   for(int i = 0; i < PositionsTotal(); i++)
     {
      if(!posinfo.SelectByIndex(i))
         continue;
      if(StringLen(items) > 0)
         items += ",";
      items += StringFormat(
         "{\"ticket\":%d,\"symbol\":\"%s\",\"direction\":\"%s\","
         "\"lots\":%.2f,\"open_price\":%.5f,\"stop_price\":%.5f,"
         "\"profit\":%.2f,\"swap\":%.2f}",
         (int)posinfo.Ticket(),
         posinfo.Symbol(),
         (posinfo.PositionType() == POSITION_TYPE_BUY ? "LONG" : "SHORT"),
         posinfo.Volume(),
         posinfo.PriceOpen(),
         posinfo.StopLoss(),
         posinfo.Profit(),
         posinfo.Swap());
     }
   return(BuildResponse(request_id, "ok", "\"positions\":[" + items + "]"));
  }

//+------------------------------------------------------------------+
string CmdQuote(const string request_id, const string symbol)
  {
   if(!SymbolSelect(symbol, true))
      return(BuildResponse(request_id, "error", "\"reason\":\"unknown symbol\""));

   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick))
      return(BuildResponse(request_id, "error", "\"reason\":\"no tick\""));

   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   string data = StringFormat("\"bid\":%.*f,\"ask\":%.*f",
                              digits, tick.bid, digits, tick.ask);
   return(BuildResponse(request_id, "ok", data));
  }

//+------------------------------------------------------------------+
//| Open with the stop attached in the SAME order_send. The stop is  |
//| mandatory: refuse rather than open an unprotected position.      |
//+------------------------------------------------------------------+
string CmdOpen(const string request_id, const string raw)
  {
   string symbol     = JsonGetString(raw, "symbol");
   string direction  = JsonGetString(raw, "direction");
   double lots       = JsonGetDouble(raw, "lots");
   double stop_price = JsonGetDouble(raw, "stop_price");
   string comment    = JsonGetString(raw, "comment");

   if(stop_price <= 0.0)
      return(BuildResponse(request_id, "rejected",
                           "\"reason\":\"no stop attached\""));
   if(lots <= 0.0)
      return(BuildResponse(request_id, "rejected",
                           "\"reason\":\"invalid lot size\""));
   if(!SymbolSelect(symbol, true))
      return(BuildResponse(request_id, "rejected",
                           "\"reason\":\"unknown symbol\""));

   // Broker minimum stop distance. Rejecting here is deliberate: silently
   // widening the stop would break the risk maths on the Python side,
   // which derived position size FROM this exact distance.
   MqlTick tick;
   SymbolInfoTick(symbol, tick);
   double entry     = (direction == "LONG") ? tick.ask : tick.bid;
   long   stops_lvl = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double point     = SymbolInfoDouble(symbol, SYMBOL_POINT);
   double min_dist  = stops_lvl * point;

   if(MathAbs(entry - stop_price) < min_dist)
      return(BuildResponse(request_id, "rejected",
             StringFormat("\"reason\":\"stop within broker minimum %.1f points\"",
                          (double)stops_lvl)));

   trade.SetExpertMagicNumber(20260820);
   bool ok = (direction == "LONG")
             ? trade.Buy(lots,  symbol, 0.0, stop_price, 0.0, comment)
             : trade.Sell(lots, symbol, 0.0, stop_price, 0.0, comment);

   if(!ok)
      return(BuildResponse(request_id, "rejected",
             StringFormat("\"reason\":\"order_send failed retcode=%d\"",
                          trade.ResultRetcode())));

   ulong  ticket = trade.ResultOrder();
   double fill   = trade.ResultPrice();

   string data = StringFormat(
      "\"ticket\":%d,\"fill_price\":%.5f,\"stop_price\":%.5f,\"lots\":%.2f",
      (int)ticket, fill, stop_price, lots);

   RecordProcessed(request_id, data);
   return(BuildResponse(request_id, "ok", data));
  }

//+------------------------------------------------------------------+
string CmdClose(const string request_id, const string raw)
  {
   ulong ticket = (ulong)JsonGetDouble(raw, "ticket");

   if(!posinfo.SelectByTicket(ticket))
      return(BuildResponse(request_id, "rejected",
                           "\"reason\":\"no such position\""));

   if(!trade.PositionClose(ticket))
      return(BuildResponse(request_id, "rejected",
             StringFormat("\"reason\":\"close failed retcode=%d\"",
                          trade.ResultRetcode())));

   string data = StringFormat("\"ticket\":%d,\"close_price\":%.5f",
                              (int)ticket, trade.ResultPrice());
   RecordProcessed(request_id, data);
   return(BuildResponse(request_id, "ok", data));
  }

//+------------------------------------------------------------------+
string CmdModifyStop(const string request_id, const string raw)
  {
   ulong  ticket     = (ulong)JsonGetDouble(raw, "ticket");
   double stop_price = JsonGetDouble(raw, "stop_price");

   if(!posinfo.SelectByTicket(ticket))
      return(BuildResponse(request_id, "rejected",
                           "\"reason\":\"no such position\""));

   if(!trade.PositionModify(ticket, stop_price, 0.0))
      return(BuildResponse(request_id, "rejected",
             StringFormat("\"reason\":\"modify failed retcode=%d\"",
                          trade.ResultRetcode())));

   string data = StringFormat("\"ticket\":%d,\"stop_price\":%.5f",
                              (int)ticket, stop_price);
   RecordProcessed(request_id, data);
   return(BuildResponse(request_id, "ok", data));
  }

//+------------------------------------------------------------------+
string BuildResponse(const string request_id, const string status,
                     const string data_fields)
  {
   return(StringFormat(
      "{\"v\":1,\"request_id\":\"%s\",\"status\":\"%s\",\"data\":{%s}}\n",
      request_id, status, data_fields));
  }

//+------------------------------------------------------------------+
//| TODO — implement for the chosen transport.                       |
//|                                                                  |
//| MQL5 sockets are client-only, so either invert the direction (EA |
//| dials out to a Python listener) or use a listening-socket DLL.   |
//| The protocol is identical either way.                            |
//|                                                                  |
//| BridgeReceive: return one complete newline-terminated frame, or  |
//| "" if nothing is pending. Must not block longer than the timer.  |
//| BridgeSend: write one frame.                                     |
//+------------------------------------------------------------------+
string BridgeReceive() { return(""); }
void   BridgeSend(const string payload) { }

//+------------------------------------------------------------------+
//| Minimal JSON field readers.                                      |
//|                                                                  |
//| The protocol is fixed, flat and machine-generated, so full        |
//| parsing is unnecessary. These do NOT handle nested objects,      |
//| arrays or escaped quotes — if the protocol grows, replace them   |
//| with a real parser rather than extending these.                  |
//+------------------------------------------------------------------+
string JsonGetString(const string json, const string key)
  {
   string needle = "\"" + key + "\":\"";
   int start = StringFind(json, needle);
   if(start < 0)
      return("");
   start += StringLen(needle);
   int end = StringFind(json, "\"", start);
   if(end < 0)
      return("");
   return(StringSubstr(json, start, end - start));
  }

double JsonGetDouble(const string json, const string key)
  {
   string needle = "\"" + key + "\":";
   int start = StringFind(json, needle);
   if(start < 0)
      return(0.0);
   start += StringLen(needle);
   int end = start;
   while(end < StringLen(json))
     {
      ushort ch = StringGetCharacter(json, end);
      if((ch >= '0' && ch <= '9') || ch == '.' || ch == '-' || ch == '+'
         || ch == 'e' || ch == 'E')
         end++;
      else
         break;
     }
   if(end == start)
      return(0.0);
   return(StringToDouble(StringSubstr(json, start, end - start)));
  }
//+------------------------------------------------------------------+
