export const initialChatState = {
  datasetId: null,
  messages: [],
  pending: false,
}

export function chatReducer(state, action) {
  switch (action.type) {
    case 'dataset_changed':
      if (state.datasetId === action.datasetId) return state
      return { datasetId: action.datasetId, messages: [], pending: false }
    case 'request_started':
      if (state.pending) return state
      return { ...state, pending: true }
    case 'request_succeeded':
      return { ...state, pending: false, messages: [...state.messages, action.message] }
    case 'request_failed':
      return { ...state, pending: false, messages: [...state.messages, action.message] }
    case 'clear':
      return { ...state, messages: [] }
    default:
      return state
  }
}
